"""物体検出を実行するクラス."""

from pathlib import Path
from typing import Any

import torch
import torchvision
from PIL import Image
from transformers import RTDetrImageProcessor

from pochidetection.inference.pytorch_backend import PyTorchBackend
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.models import RTDetrModel
from pochidetection.scripts.rtdetr.inference.detection import Detection, OutputWrapper


class Detector:
    """物体検出を実行.

    Attributes:
        _processor: 画像前処理プロセッサ.
        _backend: 推論バックエンド.
        _device: 実行デバイス.
        _threshold: 検出信頼度閾値.
        _nms_iou_threshold: NMS の IoU 閾値.
        _use_fp16: FP16 推論を使用するか.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        device: str = "cuda",
        threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        use_fp16: bool = False,
        backend: IInferenceBackend | None = None,
        processor: Any | None = None,
    ) -> None:
        """Detectorを初期化.

        Args:
            model_path: モデルディレクトリのパス. backend と processor を省略する場合は必須.
            device: 実行デバイス.
            threshold: 検出信頼度閾値.
            nms_iou_threshold: NMS の IoU 閾値.
            use_fp16: FP16 推論を使用するか. CUDA デバイスでのみ有効.
            backend: 推論バックエンドのインスタンス (DI用). 指定時は processor も必須.
            processor: 画像前処理プロセッサのインスタンス (DI用). 指定時は backend も必須.
        """
        if (backend is None) != (processor is None):
            msg = "backend と processor は両方指定するか, 両方省略する必要があります."
            raise ValueError(msg)

        if backend is None and model_path is None:
            msg = "backend と processor を省略する場合, model_path は必須です."
            raise ValueError(msg)
        self._device = device
        self._threshold = threshold
        self._nms_iou_threshold = nms_iou_threshold
        self._use_fp16 = use_fp16 and device == "cuda"

        if processor is not None:
            self._processor = processor
        else:
            assert model_path is not None
            self._processor = RTDetrImageProcessor.from_pretrained(model_path)

        if backend is not None:
            self._backend = backend
        else:
            assert model_path is not None
            # TODO: モデル初期化とバックエンド生成は外部から完全に分離するように将来リファクタ推奨
            model = RTDetrModel(str(model_path))
            model.to(device)
            model.eval()

            if self._use_fp16:
                model.half()

            self._backend = PyTorchBackend(model)

    def detect(self, image: Image.Image) -> list[Detection]:
        """画像から物体を検出.

        Args:
            image: 入力画像 (PIL Image).

        Returns:
            検出結果のリスト.
        """
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        if self._use_fp16:
            inputs = {
                k: v.half() if v.is_floating_point() else v for k, v in inputs.items()
            }

        with torch.no_grad():
            pred_logits, pred_boxes = self._backend.infer(inputs)
            self._backend.synchronize()

        # 実装依存を解消するため, HFが期待するラッパーオブジェクトを作成
        outputs = OutputWrapper(logits=pred_logits, pred_boxes=pred_boxes)

        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([image.size[::-1]]),  # (height, width)
            threshold=self._threshold,
        )[0]

        keep = torchvision.ops.nms(
            results["boxes"], results["scores"], self._nms_iou_threshold
        )
        results = {k: v[keep] for k, v in results.items()}

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
