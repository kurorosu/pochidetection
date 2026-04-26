"""E2E 推論パイプライン."""

from typing import Any, Literal, Protocol

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import v2

from pochidetection.core.detection import Detection, OutputWrapper
from pochidetection.core.letterbox import (
    LetterboxParams,
    apply_letterbox,
    compute_letterbox_params,
)
from pochidetection.core.preprocess import gpu_preprocess_tensor
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.interfaces.pipeline import IDetectionPipeline, ImageInput
from pochidetection.utils import PhasedTimer
from pochidetection.utils.device import is_fp16_available


class RTDetrPostProcessor(Protocol):
    """RT-DETR 後処理 protocol.

    `RTDetrPipeline` が依存する HF processor の唯一のメソッドを表す.
    Why: 本物の `RTDetrImageProcessor` を直接型注釈すると, テスト用 dummy
    processor (HF の重い依存を避けるため自前実装) が structural subtype として
    通らない. 必要なメソッドだけを Protocol で表現する.
    """

    def post_process_object_detection(
        self,
        outputs: Any,
        target_sizes: Any,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """検出結果を target_sizes 座標系に変換し threshold で filter する."""
        ...


class RTDetrPipeline(
    IDetectionPipeline[
        tuple[dict[str, torch.Tensor], "LetterboxParams | None"],
        tuple[torch.Tensor, torch.Tensor],
    ]
):
    """E2E 推論パイプライン.

    前処理・推論・後処理を明示的に分離し,
    PhasedTimer によるフェーズ別プロファイリングを提供する.

    Note:
        request-scoped state (letterbox の幾何パラメータ) は ``run()`` 内で
        call stack に保持し preprocess → postprocess に明示的に渡す. インスタンス
        属性には置かないため, 同一 pipeline インスタンスを複数 thread から
        ``run()`` で呼び出しても letterbox 逆変換が混線しない.

        ``_gpu_input_buffer`` は buffer 再利用による allocation コスト削減が目的の
        pipeline-scoped state. マルチ thread で同一 pipeline を共有すると偶発的に
        他 request のピクセル値で推論が走る race が成立するため, WebAPI 等の並行
        実行環境では **別 pipeline インスタンスを使う** 運用にする.

    Attributes:
        _backend: 推論バックエンド.
        _processor: 後処理用プロセッサ.
        _transform: 前処理用 torchvision v2 Transform.
        _device: 実行デバイス.
        _threshold: 検出信頼度閾値.
        _nms_iou_threshold: NMS の IoU 閾値.
        _use_fp16: FP16 推論を使用するか.
        _phased_timer: フェーズ別タイマー.
    """

    def __init__(
        self,
        backend: IInferenceBackend[tuple[torch.Tensor, torch.Tensor]],
        processor: RTDetrPostProcessor,
        transform: v2.Compose,
        device: str = "cuda",
        threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        use_fp16: bool = False,
        phased_timer: PhasedTimer | None = None,
        pipeline_mode: Literal["cpu", "gpu"] = "cpu",
        image_size: tuple[int, int] | None = None,
        letterbox: bool = False,
    ) -> None:
        """初期化.

        Args:
            backend: 推論バックエンドのインスタンス.
            processor: 後処理用プロセッサのインスタンス.
            transform: 前処理用 torchvision v2 Transform.
            device: 実行デバイス.
            threshold: 検出信頼度閾値.
            nms_iou_threshold: NMS の IoU 閾値.
            use_fp16: FP16 推論を使用するか. CUDA デバイスでのみ有効.
            phased_timer: フェーズ別タイマー. None の場合は計測しない.
            pipeline_mode: preprocess 経路 ('cpu' or 'gpu').
                'gpu' は uint8 H2D + GPU 上 float32/255 + 入力バッファ再利用で
                preprocess を高速化する. resolve_pipeline_mode() で解決済みの値.
            image_size: リサイズ先の (height, width).
                ``pipeline_mode='gpu'`` または ``letterbox=True`` の場合に必須.
                それ以外では未使用.
            letterbox: True で preprocess / postprocess に letterbox (アスペクト比
                維持 + padding) を適用し, 学習時前処理と分布を揃える. ``image_size``
                が必須. False (既定) で従来の単純 resize + 元画像サイズでの bbox
                スケーリングに戻る.

        Raises:
            ValueError: phased_timer に必須フェーズが含まれていない場合,
                または pipeline_mode='gpu' / letterbox=True で image_size が None
                の場合.
        """
        super().__init__()
        self._validate_phased_timer(phased_timer)

        if pipeline_mode == "gpu" and image_size is None:
            raise ValueError(
                "pipeline_mode='gpu' requires image_size=(H, W) to be provided"
            )
        if letterbox and image_size is None:
            raise ValueError("letterbox=True requires image_size=(H, W) to be provided")

        self._backend = backend
        self._processor = processor
        self._transform = transform
        self._device = device
        self._threshold = threshold
        self._nms_iou_threshold = nms_iou_threshold
        self._use_fp16 = is_fp16_available(use_fp16, device)
        self._pipeline_mode: Literal["cpu", "gpu"] = pipeline_mode
        self._target_hw: tuple[int, int] | None = image_size
        self._letterbox = letterbox
        self._gpu_input_buffer: torch.Tensor | None = None
        self._init_cuda_events(device)

    def preprocess(
        self, image: ImageInput
    ) -> tuple[dict[str, torch.Tensor], LetterboxParams | None]:
        """画像を前処理し, モデル入力テンソルと letterbox params を返す.

        pipeline_mode='gpu' 時は GPU 経路 (uint8 H2D + GPU 上 float32/255 +
        バッファ再利用), 'cpu' 時は従来 PIL + torchvision v2 Compose.
        ``letterbox=True`` の場合は両経路とも ``apply_letterbox`` で
        アスペクト比維持 + padding を施し, 幾何パラメータをタプルの 2 要素目で
        返す. 呼び出し側 (``run()``) はこの params を postprocess にそのまま渡し,
        bbox を元画像座標に逆変換する. 返却された params は request-scoped であり,
        インスタンス属性を介さないため thread-safe.

        Args:
            image: 入力画像 (PIL Image または numpy RGB 配列).

        Returns:
            ``(inputs, letterbox_params)`` のタプル. ``inputs`` は
            ``{"pixel_values": (1, C, H, W)}`` 形式の辞書. ``letterbox_params``
            は letterbox 有効時の幾何パラメータ, 無効時は ``None``.
        """
        # letterbox 用の幾何パラメータを事前計算 (元画像サイズ必要).
        letterbox_params: LetterboxParams | None = None
        if self._letterbox and self._target_hw is not None:
            if isinstance(image, np.ndarray):
                src_h, src_w = image.shape[:2]
            else:
                src_w, src_h = image.size
            letterbox_params = compute_letterbox_params((src_h, src_w), self._target_hw)

        if self._pipeline_mode == "gpu":
            # Why: np.asarray(PIL.Image) は read-only な配列を返し,
            # torch.from_numpy() で writable tensor を作る際に警告が出る. np.array()
            # で writable copy を作っておく.
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            return self._preprocess_gpu(image_np, letterbox_params), letterbox_params

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if letterbox_params is not None:
            image = apply_letterbox(image, letterbox_params, pad_value=0)
        pixel_values = self._transform(image).unsqueeze(0).to(self._device)

        if self._use_fp16:
            pixel_values = pixel_values.half()

        return {"pixel_values": pixel_values}, letterbox_params

    def _preprocess_gpu(
        self,
        image_np: np.ndarray,
        letterbox_params: LetterboxParams | None,
    ) -> dict[str, torch.Tensor]:
        """GPU 経路の前処理.

        ``gpu_preprocess_tensor`` ヘルパーに委譲し, 戻り値を HF 入力形式の
        dict でラップする. ヘルパー内部で uint8 → float32 キャスト + H2D 転送 +
        ``/255`` で ``[0, 1]`` 正規化を行う. バッファ再利用の state は本クラスが
        保持する. ``letterbox_params`` を渡すとアスペクト比維持 + padding を適用する.

        Args:
            image_np: RGB uint8 numpy 配列, 形状 (H, W, 3), dtype ``uint8``,
                値域 ``[0, 255]``.
            letterbox_params: letterbox 幾何パラメータ. ``None`` の場合は単純
                resize へフォールバック.

        Returns:
            HF モデル入力の辞書. キーと値は以下:
                - ``pixel_values``: 推論入力テンソル, 形状 ``(1, 3, H, W)``,
                  device は ``self._device`` (``cuda`` / ``cpu``),
                  dtype は ``use_fp16=True`` なら ``float16``, それ以外は
                  ``float32``, 値域 ``[0, 1]``. ``H``, ``W`` は
                  ``self._target_hw`` (``__init__`` で渡された ``image_size``).
        """
        assert self._target_hw is not None  # __init__ で検証済み
        pixel_values, self._gpu_input_buffer = gpu_preprocess_tensor(
            image_np=image_np,
            target_hw=self._target_hw,
            device=self._device,
            input_buffer=self._gpu_input_buffer,
            use_fp16=self._use_fp16,
            letterbox_params=letterbox_params,
        )
        return {"pixel_values": pixel_values}

    def infer(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """推論を実行.

        torch.no_grad() コンテキストで推論し, デバイス同期を行う.

        Args:
            inputs: 前処理済みのモデル入力テンソル.

        Returns:
            pred_logits と pred_boxes のタプル.
        """
        with torch.no_grad():
            pred_logits, pred_boxes = self._backend.infer(inputs)
        return pred_logits, pred_boxes

    def postprocess(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        image_size: tuple[int, int],
        threshold: float | None = None,
        letterbox_params: LetterboxParams | None = None,
    ) -> list[Detection]:
        """後処理. モデル出力を検出結果に変換する.

        letterbox 有効時は ``target_sizes`` を letterbox 後の入力サイズ
        (``self._target_hw``) にして HF から letterbox pixel 座標の bbox を取得し,
        呼出し元から渡された ``letterbox_params`` で元画像座標に逆変換する
        (``(box - pad) / scale``). letterbox 無効時は ``target_sizes`` を元画像
        サイズにして HF に直接スケーリングさせる (従来挙動).

        Args:
            pred_logits: 予測ロジット.
            pred_boxes: 予測ボックス.
            image_size: 元画像の (width, height). PIL Image.size 形式.
                letterbox 逆変換後の出力座標系はこのサイズ基準になる.
            threshold: スコア閾値を request 単位で上書きする値. ``None`` の場合は
                ``__init__`` で渡された ``self._threshold`` を使用する.
            letterbox_params: ``preprocess`` が返した幾何パラメータ. 同一 request
                内の逆変換にのみ使う request-scoped 値. ``None`` なら従来の単純
                スケーリング経路.

        Returns:
            元画像座標系での検出結果のリスト (xyxy ピクセル).
        """
        outputs = OutputWrapper(logits=pred_logits, pred_boxes=pred_boxes)

        if letterbox_params is not None and self._target_hw is not None:
            target_h, target_w = self._target_hw
            target_sizes = torch.tensor([[target_h, target_w]])
        else:
            # image_size は (width, height) なので (height, width) に変換
            target_sizes = torch.tensor([image_size[::-1]])

        effective_threshold = self._threshold if threshold is None else threshold
        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=effective_threshold,
        )[0]

        keep = torchvision.ops.nms(
            results["boxes"], results["scores"], self._nms_iou_threshold
        )
        results = {k: v[keep] for k, v in results.items()}

        boxes = results["boxes"]
        if letterbox_params is not None:
            # letterbox pixel 座標 → 元画像座標へ逆変換.
            # box - (pad_left, pad_top, pad_left, pad_top) then / scale.
            pad = torch.tensor(
                [
                    letterbox_params.pad_left,
                    letterbox_params.pad_top,
                    letterbox_params.pad_left,
                    letterbox_params.pad_top,
                ],
                dtype=boxes.dtype,
                device=boxes.device,
            )
            boxes = (boxes - pad) / letterbox_params.scale

        return [
            Detection(
                box=box.tolist(),
                score=score.item(),
                label=label.item(),
            )
            for score, label, box in zip(results["scores"], results["labels"], boxes)
        ]

    def run(
        self, image: ImageInput, *, threshold: float | None = None
    ) -> list[Detection]:
        """E2E 実行. preprocess → infer → postprocess を順に実行する.

        PhasedTimer が設定されている場合, 各フェーズを個別に計測する.

        Args:
            image: 入力画像 (PIL Image または numpy RGB 配列).
            threshold: 検出信頼度の下限しきい値. ``None`` の場合は ``__init__``
                で渡された値を使用する.

        Returns:
            検出結果のリスト.
        """
        if isinstance(image, np.ndarray):
            image_size = (image.shape[1], image.shape[0])
        else:
            image_size = image.size

        with self._measure("preprocess"):
            inputs, letterbox_params = self.preprocess(image)
        with self._measure("inference"), self._measure_inference_gpu():
            pred_logits, pred_boxes = self.infer(inputs)
        with self._measure("postprocess"):
            detections = self.postprocess(
                pred_logits,
                pred_boxes,
                image_size,
                threshold=threshold,
                letterbox_params=letterbox_params,
            )

        return detections
