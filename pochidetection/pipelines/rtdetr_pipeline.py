"""E2E 推論パイプライン."""

from typing import Literal

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import v2
from transformers import RTDetrImageProcessor

from pochidetection.core.detection import Detection, OutputWrapper
from pochidetection.core.preprocess import gpu_preprocess_tensor
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.interfaces.pipeline import IDetectionPipeline, ImageInput
from pochidetection.utils import PhasedTimer
from pochidetection.utils.device import is_fp16_available


class RTDetrPipeline(
    IDetectionPipeline[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
):
    """E2E 推論パイプライン.

    前処理・推論・後処理を明示的に分離し,
    PhasedTimer によるフェーズ別プロファイリングを提供する.

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
        processor: RTDetrImageProcessor,
        transform: v2.Compose,
        device: str = "cuda",
        threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        use_fp16: bool = False,
        phased_timer: PhasedTimer | None = None,
        pipeline_mode: Literal["cpu", "gpu"] = "cpu",
        image_size: tuple[int, int] | None = None,
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
            image_size: GPU 経路用のリサイズ先 (height, width).
                pipeline_mode='gpu' の場合に必須. 'cpu' 時は未使用.

        Raises:
            ValueError: phased_timer に必須フェーズが含まれていない場合,
                または pipeline_mode='gpu' で image_size が None の場合.
        """
        self._validate_phased_timer(phased_timer)

        if pipeline_mode == "gpu" and image_size is None:
            raise ValueError(
                "pipeline_mode='gpu' requires image_size=(H, W) to be provided"
            )

        self._backend = backend
        self._processor = processor
        self._transform = transform
        self._device = device
        self._threshold = threshold
        self._nms_iou_threshold = nms_iou_threshold
        self._use_fp16 = is_fp16_available(use_fp16, device)
        self._pipeline_mode: Literal["cpu", "gpu"] = pipeline_mode
        self._target_hw: tuple[int, int] | None = image_size
        self._gpu_input_buffer: torch.Tensor | None = None
        self._init_cuda_events(device)

    def preprocess(self, image: ImageInput) -> dict[str, torch.Tensor]:
        """画像を前処理し, モデル入力テンソルを返す.

        pipeline_mode='gpu' 時は GPU 経路 (uint8 H2D + GPU 上 float32/255 +
        バッファ再利用), 'cpu' 時は従来 PIL + torchvision v2 Compose.

        Args:
            image: 入力画像 (PIL Image または numpy RGB 配列).

        Returns:
            モデル入力テンソルの辞書.
        """
        if self._pipeline_mode == "gpu":
            # Why: np.asarray(PIL.Image) は read-only な配列を返し,
            # torch.from_numpy() で writable tensor を作る際に警告が出る. np.array()
            # で writable copy を作っておく.
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            return self._preprocess_gpu(image_np)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        pixel_values = self._transform(image).unsqueeze(0).to(self._device)

        if self._use_fp16:
            pixel_values = pixel_values.half()

        return {"pixel_values": pixel_values}

    def _preprocess_gpu(self, image_np: np.ndarray) -> dict[str, torch.Tensor]:
        """GPU 経路の前処理.

        ``gpu_preprocess_tensor`` ヘルパーに委譲し, 戻り値を HF 入力形式の
        dict でラップする. ヘルパー内部で uint8 → float32 キャスト + H2D 転送 +
        ``/255`` で ``[0, 1]`` 正規化を行う. バッファ再利用の state は本クラスが
        保持する.

        Args:
            image_np: RGB uint8 numpy 配列, 形状 (H, W, 3), dtype ``uint8``,
                値域 ``[0, 255]``.

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
    ) -> list[Detection]:
        """後処理. モデル出力を検出結果に変換する.

        Args:
            pred_logits: 予測ロジット.
            pred_boxes: 予測ボックス.
            image_size: (width, height). PIL Image.size 形式.
                内部で (height, width) に変換して HF に渡す.

        Returns:
            検出結果のリスト.
        """
        outputs = OutputWrapper(logits=pred_logits, pred_boxes=pred_boxes)

        # image_size は (width, height) なので (height, width) に変換
        target_sizes = torch.tensor([image_size[::-1]])

        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self._threshold,
        )[0]

        keep = torchvision.ops.nms(
            results["boxes"], results["scores"], self._nms_iou_threshold
        )
        results = {k: v[keep] for k, v in results.items()}

        return [
            Detection(
                box=box.tolist(),
                score=score.item(),
                label=label.item(),
            )
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            )
        ]

    def run(self, image: ImageInput) -> list[Detection]:
        """E2E 実行. preprocess → infer → postprocess を順に実行する.

        PhasedTimer が設定されている場合, 各フェーズを個別に計測する.

        Args:
            image: 入力画像 (PIL Image または numpy RGB 配列).

        Returns:
            検出結果のリスト.
        """
        if isinstance(image, np.ndarray):
            image_size = (image.shape[1], image.shape[0])
        else:
            image_size = image.size

        with self._measure("preprocess"):
            inputs = self.preprocess(image)
        with self._measure("inference"), self._measure_inference_gpu():
            pred_logits, pred_boxes = self.infer(inputs)
        with self._measure("postprocess"):
            detections = self.postprocess(pred_logits, pred_boxes, image_size)

        return detections
