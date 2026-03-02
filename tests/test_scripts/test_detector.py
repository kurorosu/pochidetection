"""Detector のテスト."""

from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from transformers import RTDetrImageProcessor

from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.scripts.rtdetr.inference.detection import (
    Detection,
    OutputWrapper,
)
from pochidetection.scripts.rtdetr.inference.detector import Detector


class TestDetection:
    """Detection dataclass のテスト."""

    def test_detection_stores_values(self) -> None:
        """Detection が値を正しく保持することを確認."""
        detection = Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.95, label=1)

        assert detection.box == [10.0, 20.0, 30.0, 40.0]
        assert detection.score == 0.95
        assert detection.label == 1

    def test_detection_equality(self) -> None:
        """同じ値の Detection が等しいことを確認."""
        d1 = Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.95, label=1)
        d2 = Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.95, label=1)

        assert d1 == d2

    def test_detection_inequality(self) -> None:
        """異なる値の Detection が等しくないことを確認."""
        d1 = Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.95, label=1)
        d2 = Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.80, label=1)

        assert d1 != d2

    def test_detection_box_format(self) -> None:
        """Detection の box が [x1, y1, x2, y2] 形式であることを確認."""
        detection = Detection(box=[0.0, 0.0, 100.0, 200.0], score=0.5, label=0)

        assert len(detection.box) == 4
        x1, y1, x2, y2 = detection.box
        assert x2 > x1
        assert y2 > y1


class DummyBackend(IInferenceBackend):
    """テスト用のダミー推論バックエンド."""

    def __init__(self) -> None:
        self.infer_called = False
        self.synchronize_called = False

    def infer(self, inputs: Any) -> tuple[Any, Any]:
        """ダミー推論."""
        self.infer_called = True
        return torch.zeros((1, 100, 2)), torch.zeros((1, 100, 4))

    def synchronize(self) -> None:
        """ダミー同期."""
        self.synchronize_called = True


class DummyProcessor:
    """テスト用のダミープロセッサ."""

    def __call__(self, images: Any, return_tensors: str) -> dict[str, Any]:
        """ダミー前処理."""
        return {"pixel_values": torch.zeros((1, 3, 64, 64))}

    def post_process_object_detection(
        self, outputs: Any, target_sizes: Any, threshold: float
    ) -> list[dict[str, Any]]:
        """ダミー後処理."""
        return [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            }
        ]


class DummyProcessorWithOverlaps:
    """重複バウンディングボックスを返すダミープロセッサ."""

    def __call__(self, images: Any, return_tensors: str) -> dict[str, Any]:
        """ダミー前処理."""
        return {"pixel_values": torch.zeros((1, 3, 64, 64))}

    def post_process_object_detection(
        self, outputs: Any, target_sizes: Any, threshold: float
    ) -> list[dict[str, Any]]:
        """重複するバウンディングボックスを返す."""
        return [
            {
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([1, 1]),
                "boxes": torch.tensor(
                    [
                        [10.0, 20.0, 30.0, 40.0],
                        [11.0, 21.0, 31.0, 41.0],
                    ]
                ),
            }
        ]


class TestDetector:
    """Detector クラスのテスト."""

    def test_detect_calls_backend_infer_and_synchronize(self) -> None:
        """Detector.detect()が推論と同期を行うことを確認."""
        dummy_backend = DummyBackend()
        dummy_processor = DummyProcessor()

        detector = Detector(
            device="cpu",
            backend=dummy_backend,
            processor=dummy_processor,
        )
        dummy_image = Image.new("RGB", (64, 64))

        detections = detector.detect(dummy_image)

        assert len(detections) == 1
        assert detections[0].score == pytest.approx(0.9, rel=1e-3)
        assert detections[0].label == 1
        assert dummy_backend.infer_called
        assert dummy_backend.synchronize_called

    def test_detector_initialization_requires_model_path_without_di(self) -> None:
        """backend と processor を省略する場合に model_path が必須であることを確認."""
        with pytest.raises(
            ValueError,
            match="backend と processor を省略する場合, model_path は必須です.",
        ):
            Detector(model_path=None)

    def test_detector_initialization_requires_both_di_components(self) -> None:
        """backend と processor が片側のみ指定された場合にエラーとなることを確認."""
        dummy_backend = DummyBackend()
        dummy_processor = DummyProcessor()

        with pytest.raises(
            ValueError,
            match="backend と processor は両方指定するか, 両方省略する必要があります.",
        ):
            Detector(model_path=Path("dummy"), backend=dummy_backend, processor=None)

        with pytest.raises(
            ValueError,
            match="backend と processor は両方指定するか, 両方省略する必要があります.",
        ):
            Detector(model_path=Path("dummy"), backend=None, processor=dummy_processor)

    def test_detect_default_nms_removes_overlapping_boxes(self) -> None:
        """デフォルト NMS (IoU=0.5) で重複ボックスが除去されることを確認."""
        detector = Detector(
            device="cpu",
            backend=DummyBackend(),
            processor=DummyProcessorWithOverlaps(),
        )
        dummy_image = Image.new("RGB", (64, 64))

        detections = detector.detect(dummy_image)

        assert len(detections) == 1
        assert detections[0].score == pytest.approx(0.9, rel=1e-3)

    def test_detect_high_nms_threshold_keeps_all(self) -> None:
        """IoU 閾値 1.0 で全検出が保持されることを確認."""
        detector = Detector(
            device="cpu",
            nms_iou_threshold=1.0,
            backend=DummyBackend(),
            processor=DummyProcessorWithOverlaps(),
        )
        dummy_image = Image.new("RGB", (64, 64))

        detections = detector.detect(dummy_image)

        assert len(detections) == 2

    def test_output_wrapper_with_real_processor(self) -> None:
        """実際の RTDetrImageProcessor と OutputWrapper の互換性を確認."""
        processor = RTDetrImageProcessor()

        # バッチサイズ 1 のダミー画像サイズ
        target_sizes = [(640, 640)]

        # 予測ロジットとボックスを保持する出力をシミュレート
        num_queries = 300
        num_classes = 80
        dummy_logits = torch.randn(1, num_queries, num_classes)
        # ランダムなボックス [0, 1]
        dummy_boxes = torch.rand(1, num_queries, 4)

        wrapper = OutputWrapper(logits=dummy_logits, pred_boxes=dummy_boxes)

        results = processor.post_process_object_detection(
            wrapper,
            target_sizes=target_sizes,
            threshold=0.5,
        )
        # scores, labels, boxes のキーを持つ辞書のリストが生成されることを確認
        assert isinstance(results, list)
        assert len(results) == 1
        assert "scores" in results[0]
        assert "labels" in results[0]
        assert "boxes" in results[0]
