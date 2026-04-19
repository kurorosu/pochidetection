"""SSD 共通 E2E 推論パイプライン."""

from typing import Literal

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from pochidetection.core.detection import Detection
from pochidetection.core.preprocess import gpu_preprocess_tensor
from pochidetection.interfaces import IInferenceBackend
from pochidetection.interfaces.pipeline import IDetectionPipeline, ImageInput
from pochidetection.utils import PhasedTimer
from pochidetection.utils.device import is_fp16_available


class SsdPipeline(
    IDetectionPipeline[
        tuple[dict[str, torch.Tensor], int, int], dict[str, torch.Tensor]
    ]
):
    """SSD 共通 E2E 推論パイプライン.

    SSDLite と SSD300 の両方で使用できる.
    前処理・推論・後処理を明示的に分離し,
    PhasedTimer によるフェーズ別プロファイリングを提供する.

    Note:
        NMS は backend 側で適用済みのため,
        後処理ではスコア閾値フィルタと座標リスケールのみ行う.

    Attributes:
        _backend: 推論バックエンド.
        _transform: torchvision 前処理パイプライン.
        _image_size: リサイズ先の (height, width).
        _device: 実行デバイス.
        _threshold: 検出信頼度閾値.
        _use_fp16: FP16 推論を使用するか.
        _phased_timer: フェーズ別タイマー.
    """

    def __init__(
        self,
        backend: IInferenceBackend[dict[str, torch.Tensor]],
        transform: v2.Compose,
        image_size: tuple[int, int],
        device: str = "cuda",
        threshold: float = 0.5,
        use_fp16: bool = False,
        phased_timer: PhasedTimer | None = None,
        pipeline_mode: Literal["cpu", "gpu"] = "cpu",
    ) -> None:
        """初期化.

        Args:
            backend: 推論バックエンド (PyTorch, ONNX, または TensorRT).
            transform: torchvision 前処理パイプライン.
            image_size: リサイズ先の (height, width).
            device: 実行デバイス.
            threshold: 検出信頼度閾値.
            use_fp16: FP16 推論を使用するか. CUDA デバイスでのみ有効.
            phased_timer: フェーズ別タイマー. None の場合は計測しない.
            pipeline_mode: preprocess 経路 ('cpu' or 'gpu').
                'gpu' は uint8 H2D + GPU 上 normalize + 入力バッファ再利用で
                preprocess を高速化する. resolve_pipeline_mode() で解決済みの値.

        Raises:
            ValueError: phased_timer に必須フェーズが含まれていない場合.
        """
        self._validate_phased_timer(phased_timer)

        self._backend = backend
        self._transform = transform
        self._image_size = image_size
        self._device = device
        self._threshold = threshold
        self._use_fp16 = is_fp16_available(use_fp16, device)
        self._pipeline_mode: Literal["cpu", "gpu"] = pipeline_mode
        self._gpu_input_buffer: torch.Tensor | None = None

    def preprocess(self, image: ImageInput) -> tuple[dict[str, torch.Tensor], int, int]:
        """画像を前処理し, モデル入力辞書と元画像サイズを返す.

        pipeline_mode='gpu' 時は GPU 経路 (uint8 H2D + GPU 上 float32/255 +
        バッファ再利用), 'cpu' 時は従来 PIL + torchvision v2 Compose.

        Args:
            image: 入力画像 (PIL Image または numpy RGB 配列).

        Returns:
            (inputs, orig_w, orig_h) のタプル.
            inputs は ``{"pixel_values": (1, C, H, W)}`` 形式の辞書で,
            RTDetr と統一された backend 入力形式.
        """
        if isinstance(image, np.ndarray):
            orig_h, orig_w = image.shape[:2]
        else:
            orig_w, orig_h = image.size

        if self._pipeline_mode == "gpu":
            # Why: np.asarray(PIL.Image) は read-only な配列を返し,
            # torch.from_numpy() で writable tensor を作る際に警告が出る. np.array()
            # で writable copy を作っておく.
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            return {"pixel_values": self._preprocess_gpu(image_np)}, orig_w, orig_h

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        pixel_values = self._transform(image).unsqueeze(0).to(self._device)

        if self._use_fp16:
            pixel_values = pixel_values.half()

        return {"pixel_values": pixel_values}, orig_w, orig_h

    def _preprocess_gpu(self, image_np: np.ndarray) -> torch.Tensor:
        """GPU 経路の前処理.

        ``gpu_preprocess_tensor`` ヘルパーに委譲する. ヘルパー内部で uint8 →
        float32 キャスト + H2D 転送 + ``/255`` で ``[0, 1]`` 正規化を行う.
        バッファ再利用の state は本クラスが保持する.

        Args:
            image_np: RGB uint8 numpy 配列, 形状 (H, W, 3), dtype ``uint8``,
                値域 ``[0, 255]``.

        Returns:
            推論入力テンソル, 形状 ``(1, 3, H, W)``, device は ``self._device``
            (``cuda`` / ``cpu``), dtype は ``use_fp16=True`` なら ``float16``,
            それ以外は ``float32``, 値域 ``[0, 1]``. ``H``, ``W`` は
            ``self._image_size``.
        """
        pixel_values, self._gpu_input_buffer = gpu_preprocess_tensor(
            image_np=image_np,
            target_hw=self._image_size,
            device=self._device,
            input_buffer=self._gpu_input_buffer,
            use_fp16=self._use_fp16,
        )
        return pixel_values

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """推論を実行.

        torch.no_grad() コンテキストで backend.infer() を呼び出す.

        Args:
            inputs: 前処理済みのモデル入力辞書 (``{"pixel_values": (1, C, H, W)}``).

        Returns:
            モデル出力の予測辞書 (boxes, scores, labels).
        """
        with torch.no_grad():
            pred: dict[str, torch.Tensor] = self._backend.infer(inputs)
        return pred

    def postprocess(
        self,
        pred: dict[str, torch.Tensor],
        orig_w: int,
        orig_h: int,
    ) -> list[Detection]:
        """後処理. スコア閾値フィルタと座標リスケールを行う.

        Note:
            NMS は backend 側で適用済みのため,
            ここではスコア閾値フィルタと座標リスケールのみ行う.

        Args:
            pred: モデル出力 (boxes, scores, labels).
            orig_w: 元画像の幅.
            orig_h: 元画像の高さ.

        Returns:
            検出結果のリスト.
        """
        mask = pred["scores"] >= self._threshold
        boxes = pred["boxes"][mask]
        scores = pred["scores"][mask]
        labels = pred["labels"][mask]

        target_h, target_w = self._image_size
        scale_x = orig_w / target_w
        scale_y = orig_h / target_h
        boxes[:, 0] *= scale_x
        boxes[:, 2] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 3] *= scale_y

        return [
            Detection(
                box=box.tolist(),
                score=score.item(),
                label=label.item(),
            )
            for box, score, label in zip(boxes, scores, labels)
        ]

    def run(self, image: ImageInput) -> list[Detection]:
        """E2E 実行. preprocess → infer → postprocess を順に実行する.

        PhasedTimer が設定されている場合, 各フェーズを個別に計測する.

        Args:
            image: 入力画像 (PIL Image または numpy RGB 配列).

        Returns:
            検出結果のリスト.
        """
        with self._measure("preprocess"):
            inputs, orig_w, orig_h = self.preprocess(image)
        with self._measure("inference"), self._measure_inference_gpu():
            pred = self.infer(inputs)
        with self._measure("postprocess"):
            detections = self.postprocess(pred, orig_w, orig_h)

        return detections
