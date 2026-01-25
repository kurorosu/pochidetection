"""物体検出を実行するクラス."""

from pathlib import Path

import torch
from PIL import Image
from transformers import RTDetrImageProcessor

from pochidetection.models import RTDetrModel
from pochidetection.scripts.rtdetr.inference.detection import Detection
from pochidetection.utils import InferenceTimer


class Detector:
    """物体検出を実行.

    Attributes:
        _processor: 画像前処理プロセッサ.
        _model: 検出モデル.
        _device: 実行デバイス.
        _threshold: 検出信頼度閾値.
        _timer: 推論時間計測タイマー.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        threshold: float = 0.5,
        timer: InferenceTimer | None = None,
    ) -> None:
        """Detectorを初期化.

        Args:
            model_path: モデルディレクトリのパス.
            device: 実行デバイス.
            threshold: 検出信頼度閾値.
            timer: 推論時間計測タイマー. Noneの場合は計測しない.
        """
        self._device = device
        self._threshold = threshold
        self._timer = timer

        self._processor = RTDetrImageProcessor.from_pretrained(model_path)
        self._model = RTDetrModel(str(model_path))
        self._model.to(device)
        self._model.eval()

    def detect(self, image: Image.Image) -> list[Detection]:
        """画像から物体を検出.

        Args:
            image: 入力画像 (PIL Image).

        Returns:
            検出結果のリスト.
        """
        # 前処理
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # 推論 (時間計測)
        if self._timer is not None:
            self._timer.start()
        with torch.no_grad():
            outputs = self._model.model(**inputs)
        if self._timer is not None:
            self._timer.stop()

        # 後処理
        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([image.size[::-1]]),  # (height, width)
            threshold=self._threshold,
        )[0]

        # Detection オブジェクトに変換
        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            detections.append(
                Detection(
                    box=box.tolist(),
                    score=score.item(),
                    label=label.item(),
                )
            )

        return detections

    @property
    def timer(self) -> InferenceTimer | None:
        """推論時間計測タイマーを取得."""
        return self._timer
