"""SsdPipeline のテスト."""

import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from pochidetection.core.detection import Detection
from pochidetection.interfaces import IInferenceBackend
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.scripts.ssd.inference.ssd_pipeline import SsdPipeline
from pochidetection.utils import PhasedTimer


class DummyBackend(IInferenceBackend[dict[str, torch.Tensor]]):
    """テスト用のダミー推論バックエンド.

    SsdPyTorchBackend と同じ出力形式を返す.
    """

    def __init__(
        self, predictions: list[dict[str, torch.Tensor]] | None = None
    ) -> None:
        """初期化.

        Args:
            predictions: 返却する予測結果のリスト.
        """
        self.call_count = 0
        self._predictions = predictions or [
            {
                "boxes": torch.tensor(
                    [[10.0, 20.0, 50.0, 60.0], [100.0, 100.0, 200.0, 200.0]]
                ),
                "scores": torch.tensor([0.9, 0.3]),
                "labels": torch.tensor([0, 1]),
            }
        ]

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """ダミー推論."""
        self.call_count += 1
        return self._predictions[0]


def _make_transform(image_size: tuple[int, int] = (320, 320)) -> v2.Compose:
    """テスト用の transform を生成."""
    return v2.Compose(
        [
            v2.Resize(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def _make_pipeline(
    backend: DummyBackend | None = None,
    threshold: float = 0.5,
    image_size: tuple[int, int] = (320, 320),
    phased_timer: PhasedTimer | None = None,
) -> SsdPipeline:
    """テスト用パイプラインを生成."""
    return SsdPipeline(
        backend=backend or DummyBackend(),
        transform=_make_transform(image_size),
        image_size=image_size,
        device="cpu",
        threshold=threshold,
        phased_timer=phased_timer,
    )


class TestSsdPipelineInit:
    """SsdPipeline の初期化テスト."""

    def test_init_without_phased_timer(self) -> None:
        """PhasedTimer なしで初期化できることを確認."""
        pipeline = _make_pipeline()
        assert pipeline.phased_timer is None

    def test_init_with_valid_phased_timer(self) -> None:
        """必須フェーズを含む PhasedTimer で初期化できることを確認."""
        timer = PhasedTimer(
            phases=["preprocess", "inference", "postprocess"],
            device="cpu",
            skip_first=False,
        )
        pipeline = _make_pipeline(phased_timer=timer)
        assert pipeline.phased_timer is timer

    def test_init_with_extra_phases_allowed(self) -> None:
        """必須フェーズ + 追加フェーズの PhasedTimer が許容されることを確認."""
        timer = PhasedTimer(
            phases=["preprocess", "inference", "postprocess", "total"],
            device="cpu",
        )
        pipeline = _make_pipeline(phased_timer=timer)
        assert pipeline.phased_timer is timer

    def test_init_missing_required_phase_raises_value_error(self) -> None:
        """必須フェーズが欠けた PhasedTimer で ValueError が発生することを確認."""
        timer = PhasedTimer(
            phases=["preprocess", "inference"],
            device="cpu",
        )
        with pytest.raises(ValueError, match="missing required phases"):
            _make_pipeline(phased_timer=timer)

    def test_implements_idetection_pipeline(self) -> None:
        """SsdPipeline が IDetectionPipeline を実装していることを確認."""
        pipeline = _make_pipeline()
        assert isinstance(pipeline, IDetectionPipeline)


class TestSsdPipelineRun:
    """SsdPipeline の run テスト."""

    def test_run_returns_detections(self) -> None:
        """run() が検出結果を返すことを確認."""
        backend = DummyBackend()
        pipeline = _make_pipeline(backend=backend, threshold=0.5)
        image = Image.new("RGB", (640, 480))

        detections = pipeline.run(image)

        assert len(detections) == 1
        assert isinstance(detections[0], Detection)
        assert detections[0].score == pytest.approx(0.9, rel=1e-3)
        assert detections[0].label == 0
        assert backend.call_count == 1

    def test_run_with_phased_timer_measures_all_phases(self) -> None:
        """PhasedTimer 付き run() で全フェーズが計測されることを確認."""
        timer = PhasedTimer(
            phases=["preprocess", "inference", "postprocess"],
            device="cpu",
            skip_first=False,
        )
        pipeline = _make_pipeline(phased_timer=timer)
        image = Image.new("RGB", (64, 64))

        pipeline.run(image)

        summary = timer.summary()
        for phase in SsdPipeline.PHASES:
            assert summary[phase]["count"] == 1

    def test_run_without_phased_timer(self) -> None:
        """PhasedTimer なしでも run() が正常動作することを確認."""
        pipeline = _make_pipeline()
        image = Image.new("RGB", (64, 64))

        detections = pipeline.run(image)

        assert len(detections) == 1


class TestSsdPipelineMethods:
    """SsdPipeline の個別メソッドテスト."""

    def test_preprocess_returns_tensor_and_original_size(self) -> None:
        """preprocess() がテンソルと元画像サイズを返すことを確認."""
        pipeline = _make_pipeline(image_size=(320, 320))
        image = Image.new("RGB", (640, 480))

        pixel_values, orig_w, orig_h = pipeline.preprocess(image)

        assert isinstance(pixel_values, torch.Tensor)
        assert pixel_values.shape == (1, 3, 320, 320)
        assert orig_w == 640
        assert orig_h == 480

    def test_infer_calls_backend(self) -> None:
        """infer() がバックエンドを呼び出すことを確認."""
        backend = DummyBackend()
        pipeline = _make_pipeline(backend=backend)
        pixel_values = torch.zeros((1, 3, 320, 320))

        pred = pipeline.infer(pixel_values)

        assert backend.call_count == 1
        assert "boxes" in pred
        assert "scores" in pred
        assert "labels" in pred

    def test_postprocess_filters_by_threshold(self) -> None:
        """postprocess() がスコア閾値でフィルタリングすることを確認."""
        pipeline = _make_pipeline(threshold=0.5, image_size=(320, 320))
        pred = {
            "boxes": torch.tensor(
                [[10.0, 20.0, 50.0, 60.0], [100.0, 100.0, 200.0, 200.0]]
            ),
            "scores": torch.tensor([0.9, 0.3]),
            "labels": torch.tensor([0, 1]),
        }

        detections = pipeline.postprocess(pred, orig_w=320, orig_h=320)

        assert len(detections) == 1
        assert detections[0].score == pytest.approx(0.9, rel=1e-3)

    def test_postprocess_rescales_coordinates(self) -> None:
        """postprocess() が座標をリスケールすることを確認."""
        pipeline = _make_pipeline(threshold=0.0, image_size=(320, 320))
        pred = {
            "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }

        detections = pipeline.postprocess(pred, orig_w=640, orig_h=480)

        assert len(detections) == 1
        box = detections[0].box
        # scale_x = 640/320 = 2.0, scale_y = 480/320 = 1.5
        assert box[0] == pytest.approx(200.0, rel=1e-3)  # 100 * 2.0
        assert box[1] == pytest.approx(150.0, rel=1e-3)  # 100 * 1.5
        assert box[2] == pytest.approx(400.0, rel=1e-3)  # 200 * 2.0
        assert box[3] == pytest.approx(300.0, rel=1e-3)  # 200 * 1.5

    def test_postprocess_empty_predictions(self) -> None:
        """閾値以上の検出がない場合に空リストを返すことを確認."""
        pipeline = _make_pipeline(threshold=0.99)
        pred = {
            "boxes": torch.tensor([[10.0, 20.0, 50.0, 60.0]]),
            "scores": torch.tensor([0.5]),
            "labels": torch.tensor([0]),
        }

        detections = pipeline.postprocess(pred, orig_w=320, orig_h=320)

        assert detections == []


class TestSsdPipelineNoNms:
    """SsdPipeline が外部 NMS を適用しないことのテスト."""

    def test_no_external_nms_applied(self) -> None:
        """重複するボックスが全て保持されることを確認 (NMS は backend 側で適用済み)."""
        overlapping_predictions = [
            {
                "boxes": torch.tensor(
                    [
                        [10.0, 20.0, 30.0, 40.0],
                        [11.0, 21.0, 31.0, 41.0],
                        [100.0, 100.0, 200.0, 200.0],
                    ]
                ),
                "scores": torch.tensor([0.9, 0.8, 0.7]),
                "labels": torch.tensor([0, 0, 1]),
            }
        ]
        backend = DummyBackend(predictions=overlapping_predictions)
        pipeline = _make_pipeline(backend=backend, threshold=0.5)
        image = Image.new("RGB", (320, 320))

        detections = pipeline.run(image)

        assert len(detections) == 3
