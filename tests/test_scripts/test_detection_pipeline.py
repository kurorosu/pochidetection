"""DetectionPipeline のテスト."""

from typing import Any

import pytest
import torch
from PIL import Image

from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.scripts.rtdetr.inference.detection import Detection
from pochidetection.scripts.rtdetr.inference.detection_pipeline import (
    DetectionPipeline,
)
from pochidetection.utils import PhasedTimer


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


class TestDetectionPipelineInit:
    """DetectionPipeline の初期化テスト."""

    def test_init_without_phased_timer(self) -> None:
        """PhasedTimer なしで初期化できることを確認."""
        pipeline = DetectionPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            device="cpu",
        )
        assert pipeline.phased_timer is None

    def test_init_with_valid_phased_timer(self) -> None:
        """必須フェーズを含む PhasedTimer で初期化できることを確認."""
        timer = PhasedTimer(
            phases=["preprocess", "inference", "postprocess"],
            device="cpu",
            skip_first=False,
        )
        pipeline = DetectionPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            device="cpu",
            phased_timer=timer,
        )
        assert pipeline.phased_timer is timer

    def test_init_with_extra_phases_allowed(self) -> None:
        """必須フェーズ + 追加フェーズの PhasedTimer が許容されることを確認."""
        timer = PhasedTimer(
            phases=["preprocess", "inference", "postprocess", "total"],
            device="cpu",
        )
        pipeline = DetectionPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            device="cpu",
            phased_timer=timer,
        )
        assert pipeline.phased_timer is timer

    def test_init_missing_required_phase_raises_value_error(self) -> None:
        """必須フェーズが欠けた PhasedTimer で ValueError が発生することを確認."""
        timer = PhasedTimer(
            phases=["preprocess", "inference"],
            device="cpu",
        )
        with pytest.raises(ValueError, match="missing required phases"):
            DetectionPipeline(
                backend=DummyBackend(),
                processor=DummyProcessor(),
                device="cpu",
                phased_timer=timer,
            )


class TestDetectionPipelineRun:
    """DetectionPipeline の run テスト."""

    def test_run_returns_detections(self) -> None:
        """run() が検出結果を返すことを確認."""
        backend = DummyBackend()
        pipeline = DetectionPipeline(
            backend=backend,
            processor=DummyProcessor(),
            device="cpu",
        )
        image = Image.new("RGB", (64, 64))

        detections = pipeline.run(image)

        assert len(detections) == 1
        assert isinstance(detections[0], Detection)
        assert detections[0].score == pytest.approx(0.9, rel=1e-3)
        assert detections[0].label == 1
        assert backend.infer_called
        assert backend.synchronize_called

    def test_run_with_phased_timer_measures_all_phases(self) -> None:
        """PhasedTimer 付き run() で全フェーズが計測されることを確認."""
        timer = PhasedTimer(
            phases=["preprocess", "inference", "postprocess"],
            device="cpu",
            skip_first=False,
        )
        pipeline = DetectionPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            device="cpu",
            phased_timer=timer,
        )
        image = Image.new("RGB", (64, 64))

        pipeline.run(image)

        summary = timer.summary()
        for phase in DetectionPipeline.PHASES:
            assert summary[phase]["count"] == 1

    def test_run_without_phased_timer(self) -> None:
        """PhasedTimer なしでも run() が正常動作することを確認."""
        pipeline = DetectionPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            device="cpu",
        )
        image = Image.new("RGB", (64, 64))

        detections = pipeline.run(image)

        assert len(detections) == 1


class TestDetectionPipelineMethods:
    """DetectionPipeline の個別メソッドテスト."""

    def test_preprocess_returns_tensors(self) -> None:
        """preprocess() がテンソル辞書を返すことを確認."""
        pipeline = DetectionPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            device="cpu",
        )
        image = Image.new("RGB", (64, 64))

        inputs = pipeline.preprocess(image)

        assert isinstance(inputs, dict)
        assert "pixel_values" in inputs
        assert isinstance(inputs["pixel_values"], torch.Tensor)

    def test_infer_calls_backend(self) -> None:
        """infer() がバックエンドを呼び出すことを確認."""
        backend = DummyBackend()
        pipeline = DetectionPipeline(
            backend=backend,
            processor=DummyProcessor(),
            device="cpu",
        )
        inputs = {"pixel_values": torch.zeros((1, 3, 64, 64))}

        pred_logits, pred_boxes = pipeline.infer(inputs)

        assert backend.infer_called
        assert backend.synchronize_called
        assert pred_logits.shape == (1, 100, 2)
        assert pred_boxes.shape == (1, 100, 4)

    def test_postprocess_returns_detections(self) -> None:
        """postprocess() が Detection リストを返すことを確認."""
        pipeline = DetectionPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            device="cpu",
        )
        pred_logits = torch.zeros((1, 100, 2))
        pred_boxes = torch.zeros((1, 100, 4))

        detections = pipeline.postprocess(pred_logits, pred_boxes, (64, 64))

        assert len(detections) == 1
        assert isinstance(detections[0], Detection)
