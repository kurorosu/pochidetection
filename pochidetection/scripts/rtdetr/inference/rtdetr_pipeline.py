"""E2E 推論パイプライン."""

from typing import Any

import torch
import torchvision
from PIL import Image
from torchvision.transforms import v2

from pochidetection.core.detection import Detection, OutputWrapper
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.utils import PhasedTimer
from pochidetection.utils.device import is_fp16_available


class RTDetrPipeline(IDetectionPipeline):
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
        backend: IInferenceBackend,
        processor: Any,
        transform: v2.Compose,
        device: str = "cuda",
        threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        use_fp16: bool = False,
        phased_timer: PhasedTimer | None = None,
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

        Raises:
            ValueError: phased_timer に必須フェーズが含まれていない場合.
        """
        self._validate_phased_timer(phased_timer)

        self._backend = backend
        self._processor = processor
        self._transform = transform
        self._device = device
        self._threshold = threshold
        self._nms_iou_threshold = nms_iou_threshold
        self._use_fp16 = is_fp16_available(use_fp16, device)

    def preprocess(self, image: Image.Image) -> dict[str, torch.Tensor]:
        """画像を前処理し, モデル入力テンソルを返す.

        Args:
            image: 入力画像 (PIL Image).

        Returns:
            モデル入力テンソルの辞書.
        """
        pixel_values = self._transform(image).unsqueeze(0).to(self._device)

        if self._use_fp16:
            pixel_values = pixel_values.half()

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
            self._backend.synchronize()
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

    def run(self, image: Image.Image) -> list[Detection]:
        """E2E 実行. preprocess → infer → postprocess を順に実行する.

        PhasedTimer が設定されている場合, 各フェーズを個別に計測する.

        Args:
            image: 入力画像 (PIL Image).

        Returns:
            検出結果のリスト.
        """
        with self._measure("preprocess"):
            inputs = self.preprocess(image)
        with self._measure("inference"):
            pred_logits, pred_boxes = self.infer(inputs)
        with self._measure("postprocess"):
            detections = self.postprocess(pred_logits, pred_boxes, image.size)

        return detections
