"""E2E 推論パイプライン."""

from typing import Any

import torch
from PIL import Image

from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.scripts.rtdetr.inference.detection import Detection, OutputWrapper
from pochidetection.utils import PhasedTimer


class DetectionPipeline:
    """E2E 推論パイプライン.

    前処理・推論・後処理を明示的に分離し,
    PhasedTimer によるフェーズ別プロファイリングを提供する.

    Attributes:
        _backend: 推論バックエンド.
        _processor: 画像前処理プロセッサ.
        _device: 実行デバイス.
        _threshold: 検出信頼度閾値.
        _use_fp16: FP16 推論を使用するか.
        _phased_timer: フェーズ別タイマー.
    """

    PHASES = ["preprocess", "inference", "postprocess"]

    def __init__(
        self,
        backend: IInferenceBackend,
        processor: Any,
        device: str = "cuda",
        threshold: float = 0.5,
        use_fp16: bool = False,
        phased_timer: PhasedTimer | None = None,
    ) -> None:
        """初期化.

        Args:
            backend: 推論バックエンドのインスタンス.
            processor: 画像前処理プロセッサのインスタンス.
            device: 実行デバイス.
            threshold: 検出信頼度閾値.
            use_fp16: FP16 推論を使用するか. CUDA デバイスでのみ有効.
            phased_timer: フェーズ別タイマー. None の場合は計測しない.

        Raises:
            ValueError: phased_timer に必須フェーズが含まれていない場合.
        """
        if phased_timer is not None:
            missing = set(self.PHASES) - set(phased_timer.phases)
            if missing:
                raise ValueError(
                    f"phased_timer is missing required phases: {sorted(missing)}. "
                    f"Required: {self.PHASES}"
                )

        self._backend = backend
        self._processor = processor
        self._device = device
        self._threshold = threshold
        self._use_fp16 = use_fp16 and device == "cuda"
        self._phased_timer = phased_timer

    def preprocess(self, image: Image.Image) -> dict[str, torch.Tensor]:
        """画像を前処理し, モデル入力テンソルを返す.

        Args:
            image: 入力画像 (PIL Image).

        Returns:
            モデル入力テンソルの辞書.
        """
        batch = self._processor(images=image, return_tensors="pt")
        inputs: dict[str, torch.Tensor] = {
            k: v.to(self._device) for k, v in batch.items()
        }

        if self._use_fp16:
            inputs = {
                k: v.half() if v.is_floating_point() else v for k, v in inputs.items()
            }

        return inputs

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
        if self._phased_timer is not None:
            with self._phased_timer.measure("preprocess"):
                inputs = self.preprocess(image)
            with self._phased_timer.measure("inference"):
                pred_logits, pred_boxes = self.infer(inputs)
            with self._phased_timer.measure("postprocess"):
                detections = self.postprocess(pred_logits, pred_boxes, image.size)
        else:
            inputs = self.preprocess(image)
            pred_logits, pred_boxes = self.infer(inputs)
            detections = self.postprocess(pred_logits, pred_boxes, image.size)

        return detections

    @property
    def phased_timer(self) -> PhasedTimer | None:
        """フェーズ別タイマーを取得."""
        return self._phased_timer
