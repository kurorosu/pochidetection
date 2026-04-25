"""RTDetrPipeline のテスト."""

import warnings
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from pochidetection.core.detection import Detection
from pochidetection.interfaces import IInferenceBackend
from pochidetection.pipelines.rtdetr_pipeline import RTDetrPipeline
from pochidetection.utils import PhasedTimer

DUMMY_IMAGE_SIZE = (64, 64)

DUMMY_TRANSFORM = v2.Compose(
    [
        v2.Resize(DUMMY_IMAGE_SIZE, interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


class DummyBackend(IInferenceBackend[tuple[torch.Tensor, torch.Tensor]]):
    """テスト用のダミー推論バックエンド."""

    def __init__(self) -> None:
        self.infer_called = False

    def infer(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ダミー推論."""
        self.infer_called = True
        return torch.zeros((1, 100, 2)), torch.zeros((1, 100, 4))


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

    def test_run_on_cpu_leaves_gpu_inference_ms_none(self) -> None:
        """CPU 実行時は last_inference_gpu_ms が None のまま (CUDA Event 不発火)."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
        )
        image = Image.new("RGB", (64, 64))

        pipeline.run(image)

        assert pipeline.last_inference_gpu_ms is None


class TestRTDetrPipelineMethods:
    """RTDetrPipeline の個別メソッドテスト."""

    def test_preprocess_returns_tensors(self) -> None:
        """preprocess() がテンソル辞書 + letterbox params を返すことを確認."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
        )
        image = Image.new("RGB", (64, 64))

        inputs, letterbox_params = pipeline.preprocess(image)

        assert isinstance(inputs, dict)
        assert "pixel_values" in inputs
        assert isinstance(inputs["pixel_values"], torch.Tensor)
        assert letterbox_params is None  # letterbox=False 時は None

    def test_preprocess_output_shape(self) -> None:
        """preprocess() の出力テンソル形状を確認."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
        )
        image = Image.new("RGB", (128, 96))

        inputs, _ = pipeline.preprocess(image)

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

        inputs, _ = pipeline.preprocess(image)

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

    def test_postprocess_threshold_override_is_passed_to_processor(self) -> None:
        """postprocess() の threshold 引数が HF processor に渡される."""

        class ThresholdCapturingProcessor:
            def __init__(self) -> None:
                self.last_threshold: float | None = None

            def post_process_object_detection(
                self, outputs: Any, target_sizes: Any, threshold: float
            ) -> list[dict[str, Any]]:
                self.last_threshold = threshold
                return [
                    {
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([1]),
                        "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                    }
                ]

        processor = ThresholdCapturingProcessor()
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=processor,
            transform=DUMMY_TRANSFORM,
            device="cpu",
            threshold=0.5,
        )
        pred_logits = torch.zeros((1, 100, 2))
        pred_boxes = torch.zeros((1, 100, 4))

        pipeline.postprocess(pred_logits, pred_boxes, (64, 64), threshold=0.1)
        assert processor.last_threshold == pytest.approx(0.1)

        pipeline.postprocess(pred_logits, pred_boxes, (64, 64), threshold=None)
        assert processor.last_threshold == pytest.approx(0.5)

    def test_run_threshold_override_passes_to_processor(self) -> None:
        """run() の threshold 引数が postprocess 経由で processor に届く."""

        class ThresholdCapturingProcessor:
            def __init__(self) -> None:
                self.last_threshold: float | None = None

            def post_process_object_detection(
                self, outputs: Any, target_sizes: Any, threshold: float
            ) -> list[dict[str, Any]]:
                self.last_threshold = threshold
                return [
                    {
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([1]),
                        "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                    }
                ]

        processor = ThresholdCapturingProcessor()
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=processor,
            transform=DUMMY_TRANSFORM,
            device="cpu",
            threshold=0.5,
        )
        image = Image.new("RGB", (64, 64))

        pipeline.run(image, threshold=0.2)
        assert processor.last_threshold == pytest.approx(0.2)


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


class TestRTDetrPipelineMode:
    """RTDetrPipeline の pipeline_mode (CPU/GPU 経路切替) のテスト."""

    # pipeline_mode プロパティの default / cpu / gpu 反映テストは
    # tests/test_pipelines/test_pipeline_mode.py に parametrize で統合済み.

    def test_gpu_mode_without_image_size_raises_value_error(self) -> None:
        """pipeline_mode='gpu' で image_size 未指定なら ValueError."""
        with pytest.raises(ValueError, match="image_size=\\(H, W\\)"):
            RTDetrPipeline(
                backend=DummyBackend(),
                processor=DummyProcessor(),
                transform=DUMMY_TRANSFORM,
                device="cpu",
                pipeline_mode="gpu",
            )

    def test_gpu_preprocess_returns_correct_shape_and_range(self) -> None:
        """GPU 経路 preprocess の出力 shape と [0, 1] 範囲を確認."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
            pipeline_mode="gpu",
            image_size=DUMMY_IMAGE_SIZE,
        )
        image = np.random.randint(0, 256, size=(96, 128, 3), dtype=np.uint8)

        inputs, _ = pipeline.preprocess(image)

        assert inputs["pixel_values"].shape == (1, 3, 64, 64)
        assert inputs["pixel_values"].dtype == torch.float32
        assert inputs["pixel_values"].min() >= 0.0
        assert inputs["pixel_values"].max() <= 1.0

    def test_cpu_gpu_preprocess_numerically_close(self) -> None:
        """CPU 経路と GPU 経路で preprocess 出力が許容差内で一致する.

        PIL BILINEAR と tensor BILINEAR は完全一致しないため abs=1e-2 程度の差
        は想定内 (1-2 uint8 値の差 / 255 ≈ 0.008).
        """
        image = np.random.randint(0, 256, size=(96, 128, 3), dtype=np.uint8)

        cpu_pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
            pipeline_mode="cpu",
        )
        gpu_pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
            pipeline_mode="gpu",
            image_size=DUMMY_IMAGE_SIZE,
        )

        cpu_out = cpu_pipeline.preprocess(image)[0]["pixel_values"]
        gpu_out = gpu_pipeline.preprocess(image)[0]["pixel_values"]

        assert cpu_out.shape == gpu_out.shape
        # PIL/tensor BILINEAR の差は abs=1e-2 (≈ 2.5/255) 以内に収まる
        assert torch.allclose(cpu_out, gpu_out, atol=1e-2)

    def test_gpu_buffer_is_reused_across_calls(self) -> None:
        """同じ shape の入力で GPU buffer が再利用される (再確保しない)."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
            pipeline_mode="gpu",
            image_size=DUMMY_IMAGE_SIZE,
        )
        image = np.zeros((96, 128, 3), dtype=np.uint8)

        out1 = pipeline.preprocess(image)[0]["pixel_values"]
        buf_id_1 = id(pipeline._gpu_input_buffer)
        out2 = pipeline.preprocess(image)[0]["pixel_values"]
        buf_id_2 = id(pipeline._gpu_input_buffer)

        assert buf_id_1 == buf_id_2  # 同一インスタンス再利用
        assert out1.shape == out2.shape

    def test_gpu_preprocess_accepts_pil_image_without_warning(self) -> None:
        """GPU 経路で PIL Image 入力も正常動作し read-only numpy 警告が出ない.

        np.asarray(PIL.Image) は read-only だが np.array() で writable copy を
        作っているため torch.from_numpy() の警告が抑制される.
        """
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
            pipeline_mode="gpu",
            image_size=DUMMY_IMAGE_SIZE,
        )
        image = Image.new("RGB", (128, 96), color=(64, 128, 255))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            inputs, _ = pipeline.preprocess(image)

        assert inputs["pixel_values"].shape == (1, 3, 64, 64)
        assert all("not writable" not in str(w.message) for w in caught)


class TestRTDetrPipelineLetterbox:
    """letterbox=True 経路の preprocess / postprocess 動作を検証."""

    def test_init_raises_when_letterbox_true_without_image_size(self) -> None:
        """letterbox=True + image_size=None は ValueError."""
        with pytest.raises(ValueError, match="letterbox=True requires image_size"):
            RTDetrPipeline(
                backend=DummyBackend(),
                processor=DummyProcessor(),
                transform=DUMMY_TRANSFORM,
                device="cpu",
                letterbox=True,
            )

    def test_preprocess_returns_letterbox_params(self) -> None:
        """letterbox=True で preprocess が幾何パラメータを戻り値で返す."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
            image_size=(64, 64),
            letterbox=True,
        )
        # 横長 128x32 → 64x64: scale = min(64/32, 64/128) = 0.5, new=(16,64),
        # pad_vertical = 48 → pad_top = 24, pad_bottom = 24, pad_left = pad_right = 0
        image = Image.new("RGB", (128, 32))

        _, params = pipeline.preprocess(image)

        assert params is not None
        assert params.scale == pytest.approx(0.5)
        assert (params.new_h, params.new_w) == (16, 64)
        assert (params.pad_top, params.pad_bottom) == (24, 24)
        assert (params.pad_left, params.pad_right) == (0, 0)

    def test_preprocess_returns_none_params_when_letterbox_disabled(self) -> None:
        """letterbox=False で preprocess の params 戻り値が None."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
            letterbox=False,
        )
        image = Image.new("RGB", (128, 32))

        _, params = pipeline.preprocess(image)

        assert params is None

    def test_pipeline_has_no_last_letterbox_params_attribute(self) -> None:
        """request-scoped 化後は _last_letterbox_params インスタンス属性を持たない."""
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
            image_size=(64, 64),
            letterbox=True,
        )
        # letterbox 幾何パラメータは instance 属性ではなく call stack で扱う.
        assert not hasattr(pipeline, "_last_letterbox_params")

    def test_postprocess_reverses_letterbox_transform(self) -> None:
        """letterbox 経路では processor 出力 (letterbox pixel 座標) が元画像座標に逆変換される.

        DummyProcessor は ``[[10, 20, 30, 40]]`` を返す. target_hw=(64, 64) で
        letterbox params(scale=0.5, pad_top=24, pad_left=0) を設定した場合,
        逆変換後の bbox は ``[(10-0)/0.5, (20-24)/0.5, (30-0)/0.5, (40-24)/0.5]``
        = ``[20, -8, 60, 32]``. 元画像 128x32 の座標系.
        """
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=DummyProcessor(),
            transform=DUMMY_TRANSFORM,
            device="cpu",
            image_size=(64, 64),
            letterbox=True,
        )
        # run 経由で params を設定.
        image = Image.new("RGB", (128, 32))
        detections = pipeline.run(image)

        assert len(detections) == 1
        box = detections[0].box
        assert box[0] == pytest.approx(20.0, abs=1e-3)
        assert box[1] == pytest.approx(-8.0, abs=1e-3)
        assert box[2] == pytest.approx(60.0, abs=1e-3)
        assert box[3] == pytest.approx(32.0, abs=1e-3)

    def test_postprocess_passes_target_hw_as_target_sizes_when_letterbox(self) -> None:
        """letterbox 時は HF に target_sizes=(target_h, target_w) を渡す (捕捉)."""

        class TargetCapturingProcessor:
            def __init__(self) -> None:
                self.last_target_sizes: torch.Tensor | None = None

            def post_process_object_detection(
                self, outputs: Any, target_sizes: Any, threshold: float
            ) -> list[dict[str, Any]]:
                self.last_target_sizes = target_sizes
                return [
                    {
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([1]),
                        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    }
                ]

        processor = TargetCapturingProcessor()
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=processor,
            transform=DUMMY_TRANSFORM,
            device="cpu",
            image_size=(64, 64),
            letterbox=True,
        )
        image = Image.new("RGB", (128, 32))
        pipeline.run(image)

        assert processor.last_target_sizes is not None
        assert processor.last_target_sizes.tolist() == [[64, 64]]

    def test_letterbox_false_keeps_legacy_target_sizes(self) -> None:
        """letterbox=False は従来通り target_sizes=(orig_h, orig_w) を渡す."""

        class TargetCapturingProcessor:
            def __init__(self) -> None:
                self.last_target_sizes: torch.Tensor | None = None

            def post_process_object_detection(
                self, outputs: Any, target_sizes: Any, threshold: float
            ) -> list[dict[str, Any]]:
                self.last_target_sizes = target_sizes
                return [
                    {
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([1]),
                        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    }
                ]

        processor = TargetCapturingProcessor()
        pipeline = RTDetrPipeline(
            backend=DummyBackend(),
            processor=processor,
            transform=DUMMY_TRANSFORM,
            device="cpu",
            image_size=(64, 64),
            letterbox=False,
        )
        image = Image.new("RGB", (128, 32))
        pipeline.run(image)

        # orig (w, h) = (128, 32) → target_sizes = [[32, 128]] (H, W)
        assert processor.last_target_sizes is not None
        assert processor.last_target_sizes.tolist() == [[32, 128]]
