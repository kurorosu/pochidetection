"""RTDetrPipeline のテスト."""

from typing import Any

import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from pochidetection.core.detection import Detection
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.scripts.rtdetr.inference.rtdetr_pipeline import (
    RTDetrPipeline,
)
from pochidetection.utils import PhasedTimer

DUMMY_IMAGE_SIZE = (64, 64)

DUMMY_TRANSFORM = v2.Compose(
    [
        v2.Resize(DUMMY_IMAGE_SIZE, interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


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
    """テスト用のダミープロセッサ (後処理のみ)."""

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
    """重複バウンディングボックスを返すダミープロセッサ (後処理のみ)."""

    def post_process_object_detection(
        self, outputs: Any, target_sizes: Any, threshold: float
    ) -> list[dict[str, Any]]:
        """重複するバウンディングボックスを返す."""
        return [
            {
                "scores": torch.tensor([0.9, 0.8, 0.5]),
                "labels": torch.tensor([1, 1, 2]),
                "boxes": torch.tensor(
                    [
                        [10.0, 20.0, 30.0, 40.0],
                        [11.0, 21.0, 31.0, 41.0],
                        [100.0, 100.0, 200.0, 200.0],
                    ]
                ),
            }
        ]


class TestRTDetrPipelineInit:
    """RTDetrPipeline の初期化テスト."""

    def test_init_without_phased_timer(self) -> None:
        """PhasedTimer なしで初期化できることを確認."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
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
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
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
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
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
            RTDetrPipeline(
                backend=DummyBackend(),
                processor=DummyProcessor(),
                transform=DUMMY_TRANSFORM,
                device="cpu",
                phased_timer=timer,
            )


class TestRTDetrPipelineRun:
    """RTDetrPipeline の run テスト."""

    def test_run_returns_detections(self) -> None:
        """run() が検出結果を返すことを確認."""
        backend = DummyBackend()
        pipeline = RTDetrPipeline(
            backend=backend,
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
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
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
            phased_timer=timer,
        )
        image = Image.new("RGB", (64, 64))

        pipeline.run(image)

        summary = timer.summary()
        for phase in RTDetrPipeline.PHASES:
            assert summary[phase]["count"] == 1

    def test_run_without_phased_timer(self) -> None:
        """PhasedTimer なしでも run() が正常動作することを確認."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
        )
        image = Image.new("RGB", (64, 64))

        detections = pipeline.run(image)

        assert len(detections) == 1


class TestRTDetrPipelineMethods:
    """RTDetrPipeline の個別メソッドテスト."""

    def test_preprocess_returns_tensors(self) -> None:
        """preprocess() がテンソル辞書を返すことを確認."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
        )
        image = Image.new("RGB", (64, 64))

        inputs = pipeline.preprocess(image)

        assert isinstance(inputs, dict)
        assert "pixel_values" in inputs
        assert isinstance(inputs["pixel_values"], torch.Tensor)

    def test_preprocess_output_shape(self) -> None:
        """preprocess() の出力テンソル形状を確認."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
        )
        image = Image.new("RGB", (128, 96))

        inputs = pipeline.preprocess(image)

        assert inputs["pixel_values"].shape == (1, 3, 64, 64)
        assert inputs["pixel_values"].dtype == torch.float32

    def test_preprocess_pixel_range(self) -> None:
        """preprocess() の出力が [0, 1] 範囲であることを確認."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
        )
        image = Image.new("RGB", (64, 64), color=(128, 64, 255))

        inputs = pipeline.preprocess(image)

        assert inputs["pixel_values"].min() >= 0.0
        assert inputs["pixel_values"].max() <= 1.0

    def test_infer_calls_backend(self) -> None:
        """infer() がバックエンドを呼び出すことを確認."""
        backend = DummyBackend()
        pipeline = RTDetrPipeline(
            backend=backend,
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
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
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
        )
        pred_logits = torch.zeros((1, 100, 2))
        pred_boxes = torch.zeros((1, 100, 4))

        detections = pipeline.postprocess(pred_logits, pred_boxes, (64, 64))

        assert len(detections) == 1
        assert isinstance(detections[0], Detection)


class TestRTDetrPipelineNms:
    """RTDetrPipeline の NMS テスト."""

    def test_nms_enabled_by_default(self) -> None:
        """デフォルト (IoU=0.5) で NMS が適用され重複が除去されることを確認."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessorWithOverlaps(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
        )
        image = Image.new("RGB", (64, 64))

        detections = pipeline.run(image)

        # 重複する2つ (IoU > 0.5) のうちスコア高い方が残り, 離れた1つも残る
        assert len(detections) == 2
        scores = [d.score for d in detections]
        assert pytest.approx(0.9, rel=1e-3) in scores
        assert pytest.approx(0.5, rel=1e-3) in scores

    def test_nms_threshold_zero_suppresses_overlapping(self) -> None:
        """IoU 閾値 0.0 で IoU > 0 の重複ペアが抑制されることを確認."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessorWithOverlaps(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
            nms_iou_threshold=0.0,
        )
        image = Image.new("RGB", (64, 64))

        detections = pipeline.run(image)

        # 重複する2つのうちスコア低い方が抑制され, 離れた1つは残る
        assert len(detections) == 2
        scores = sorted([d.score for d in detections], reverse=True)
        assert scores[0] == pytest.approx(0.9, rel=1e-3)
        assert scores[1] == pytest.approx(0.5, rel=1e-3)

    def test_nms_threshold_one_keeps_all(self) -> None:
        """IoU 閾値 1.0 で全検出が保持されることを確認."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessorWithOverlaps(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
            nms_iou_threshold=1.0,
        )
        image = Image.new("RGB", (64, 64))

        detections = pipeline.run(image)

        assert len(detections) == 3
