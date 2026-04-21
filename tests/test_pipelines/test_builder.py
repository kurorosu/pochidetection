"""pipelines/builder.py のテスト."""

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.core.detection import Detection
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.interfaces.pipeline import IDetectionPipeline, ImageInput
from pochidetection.pipelines.builder import (
    ArchitectureSpec,
    BackendFactories,
    PipelineContext,
    _collect_image_files,
    _resolve_model_path,
    _run_inference,
    resolve_pipeline_mode,
    setup_pipeline,
)
from pochidetection.reporting import InferenceSaver, Visualizer
from pochidetection.utils import PhasedTimer


class TestResolveModelPath:
    """resolve_model_path のテスト."""

    def test_returns_path_when_model_dir_exists(self, tmp_path: Path) -> None:
        """model_dir が存在する場合そのパスを返すことを確認."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config: DetectionConfigDict = {"work_dir": str(tmp_path)}

        result = _resolve_model_path(config, str(model_dir))

        assert result == model_dir

    def test_returns_none_when_model_dir_not_exists(self, tmp_path: Path) -> None:
        """model_dir が存在しない場合 None を返すことを確認."""
        config: DetectionConfigDict = {"work_dir": str(tmp_path)}

        result = _resolve_model_path(config, str(tmp_path / "nonexistent"))

        assert result is None

    def test_returns_pretrained_when_no_workspaces(self, tmp_path: Path) -> None:
        """model_dir=None でワークスペースが無い場合 PRETRAINED を返すことを確認."""
        from pochidetection.pipelines.builder import PRETRAINED

        work_dir = tmp_path / "work_dirs"
        work_dir.mkdir()
        config: DetectionConfigDict = {"work_dir": str(work_dir)}

        result = _resolve_model_path(config, None)

        assert result == PRETRAINED

    def test_returns_best_from_latest_workspace(self, tmp_path: Path) -> None:
        """model_dir=None で最新ワークスペースの best を返すことを確認."""
        work_dir = tmp_path / "work_dirs"
        workspace = work_dir / "20260101_001"
        best_dir = workspace / "best"
        best_dir.mkdir(parents=True)
        config: DetectionConfigDict = {"work_dir": str(work_dir)}

        result = _resolve_model_path(config, None)

        assert result == best_dir

    def test_returns_none_when_best_not_exists(self, tmp_path: Path) -> None:
        """model_dir=None で best ディレクトリが無い場合 None を返すことを確認."""
        work_dir = tmp_path / "work_dirs"
        workspace = work_dir / "20260101_001"
        workspace.mkdir(parents=True)
        config: DetectionConfigDict = {"work_dir": str(work_dir)}

        result = _resolve_model_path(config, None)

        assert result is None


class TestCollectImageFiles:
    """collect_image_files のテスト."""

    def test_returns_none_when_dir_not_exists(self, tmp_path: Path) -> None:
        """ディレクトリが存在しない場合 None を返すことを確認."""
        result = _collect_image_files(str(tmp_path / "nonexistent"))

        assert result is None

    def test_returns_none_when_no_images(self, tmp_path: Path) -> None:
        """画像ファイルが無い場合 None を返すことを確認."""
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b")

        result = _collect_image_files(str(tmp_path))

        assert result is None

    def test_returns_image_files(self, tmp_path: Path) -> None:
        """画像ファイルを正しく収集することを確認."""
        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8")
        (tmp_path / "diagram.png").write_bytes(b"\x89PNG")
        (tmp_path / "notes.txt").write_text("not an image")

        result = _collect_image_files(str(tmp_path))

        assert result is not None
        assert len(result) == 2
        names = {f.name for f in result}
        assert names == {"photo.jpg", "diagram.png"}

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        """大文字拡張子の画像ファイルも収集することを確認."""
        (tmp_path / "photo.JPG").write_bytes(b"\xff\xd8")
        (tmp_path / "image.PNG").write_bytes(b"\x89PNG")

        result = _collect_image_files(str(tmp_path))

        assert result is not None
        assert len(result) == 2


class TestResolvePipelineMode:
    """resolve_pipeline_mode のテスト."""

    @pytest.mark.parametrize(
        ("requested", "suffix", "expected"),
        [
            (None, ".pt", "gpu"),
            (None, ".engine", "gpu"),
            (None, ".onnx", "cpu"),
            ("cpu", ".pt", "cpu"),
            ("gpu", ".pt", "gpu"),
            ("gpu", ".engine", "gpu"),
            ("cpu", ".onnx", "cpu"),
        ],
        ids=[
            "pytorch_default_gpu",
            "tensorrt_default_gpu",
            "onnx_default_cpu",
            "pytorch_explicit_cpu",
            "pytorch_explicit_gpu",
            "tensorrt_explicit_gpu",
            "onnx_explicit_cpu",
        ],
    )
    def test_resolves_mode(
        self,
        tmp_path: Path,
        requested: str | None,
        suffix: str,
        expected: str,
    ) -> None:
        """backend 種別と requested の組合せで解決後の経路名が返ることを確認."""
        model_path = tmp_path / f"model{suffix}"
        assert resolve_pipeline_mode(requested, model_path) == expected  # type: ignore[arg-type]

    def test_onnx_with_explicit_gpu_raises_value_error(self, tmp_path: Path) -> None:
        """ONNX backend + 明示 'gpu' は ValueError で起動拒否しメッセージも一致."""
        model_path = tmp_path / "model.onnx"
        with pytest.raises(ValueError, match="ONNX backend"):
            resolve_pipeline_mode("gpu", model_path)


class _StubPipeline(IDetectionPipeline[Any, Any]):
    """run() が空検出を返し, 付随する phased_timer もダミーで持つ stub."""

    def __init__(self, phased_timer: PhasedTimer) -> None:
        super().__init__()
        self._validate_phased_timer(phased_timer)

    def run(
        self, image: ImageInput, *, threshold: float | None = None
    ) -> list[Detection]:
        """空の検出リストを返す."""
        return []


def _build_stub_context(saver_base: Path) -> PipelineContext:
    """_run_inference テスト用の最小 PipelineContext を構築する."""
    timer = PhasedTimer(
        phases=["preprocess", "inference", "postprocess"],
        device="cpu",
        skip_first=False,
    )
    pipeline = _StubPipeline(timer)
    saver = InferenceSaver(saver_base)
    visualizer = Visualizer()
    return PipelineContext(
        pipeline=pipeline,
        phased_timer=timer,
        visualizer=visualizer,
        saver=saver,
        label_mapper=None,
        class_names=None,
        actual_device="cpu",
        precision="fp32",
    )


class TestRunInferenceInferDebug:
    """_run_inference の infer_debug 保存動作 (E2E 寄り)."""

    def _prepare_images(self, dir_path: Path, n: int) -> list[Path]:
        """jpg を n 枚作成してパスリストを返す."""
        dir_path.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for i in range(n):
            p = dir_path / f"img_{i:03d}.jpg"
            Image.new("RGB", (128, 32), color=(i * 60, 0, 0)).save(p, format="JPEG")
            files.append(p)
        return files

    def test_saves_first_n_images_to_inference_output_dir(self, tmp_path: Path) -> None:
        """save_count=2 なら先頭 2 枚のみ ``<output_dir>/infer_debug/`` に保存."""
        saver_base = tmp_path / "model"
        saver_base.mkdir()
        image_files = self._prepare_images(tmp_path / "images", 3)

        ctx = _build_stub_context(saver_base)
        config: DetectionConfigDict = {
            "infer_debug_save_count": 2,
            "image_size": {"height": 64, "width": 64},
            "letterbox": True,
        }

        _run_inference(image_files, ctx, config, save_crop=False)

        debug_dir = ctx.saver.output_dir / "infer_debug"
        saved = sorted(p.name for p in debug_dir.iterdir() if p.suffix == ".jpg")
        assert saved == ["infer_0000.jpg", "infer_0001.jpg"]

    def test_no_save_when_disabled(self, tmp_path: Path) -> None:
        """infer_debug_save_count=0 で ``infer_debug/`` が作られない."""
        saver_base = tmp_path / "model"
        saver_base.mkdir()
        image_files = self._prepare_images(tmp_path / "images", 2)

        ctx = _build_stub_context(saver_base)
        config: DetectionConfigDict = {
            "infer_debug_save_count": 0,
            "image_size": {"height": 64, "width": 64},
            "letterbox": True,
        }

        _run_inference(image_files, ctx, config, save_crop=False)

        assert not (ctx.saver.output_dir / "infer_debug").exists()


class _FakeBackend(IInferenceBackend[Any]):
    """最小 stub backend."""

    def infer(self, inputs: Any) -> Any:
        """空結果を返す."""
        return None


class _RecordingPipeline(IDetectionPipeline[Any, Any]):
    """setup_pipeline が渡した kwargs を ``init_kwargs`` に記録する stub pipeline.

    ArchitectureSpec 経由の kwargs 組立が意図通り (processor / image_size /
    nms_iou_threshold 等) になっているかを検証する.
    """

    PHASES = ["preprocess", "inference", "postprocess"]

    init_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        """受け取った kwargs を instance 属性と class 属性両方に記録."""
        super().__init__()
        phased_timer = kwargs.get("phased_timer")
        self._validate_phased_timer(phased_timer)
        self.received_kwargs = kwargs
        # class 属性経由で spec-level アサートを可能にする.
        type(self).init_kwargs = kwargs

    def run(
        self, image: ImageInput, *, threshold: float | None = None
    ) -> list[Detection]:
        """空リストを返す."""
        return []


class TestSetupPipeline:
    """ArchitectureSpec + setup_pipeline の統合テスト."""

    def _make_config(self, tmp_path: Path) -> DetectionConfigDict:
        return {
            "work_dir": str(tmp_path),
            "device": "cpu",
            "use_fp16": False,
            "infer_score_threshold": 0.5,
            "nms_iou_threshold": 0.4,
            "cudnn_benchmark": False,
            "image_size": {"height": 320, "width": 320},
            "letterbox": True,
            "pipeline_mode": "cpu",
            "class_names": ["a"],
            "num_classes": 1,
        }

    def test_builds_context_with_common_kwargs(self, tmp_path: Path) -> None:
        """共通 kwargs (backend / threshold / letterbox 等) が pipeline に渡る."""
        _RecordingPipeline.init_kwargs = None
        model_path = tmp_path / "model"
        model_path.mkdir()
        config = self._make_config(tmp_path)

        spec = ArchitectureSpec(
            pipeline_cls=_RecordingPipeline,
            backends=BackendFactories(
                pytorch=lambda p, d, fp16: _FakeBackend(),
                onnx=lambda p, d: _FakeBackend(),
                tensorrt=lambda p: _FakeBackend(),
                trt_available=False,
            ),
        )

        ctx = setup_pipeline(spec, config, model_path)

        kwargs = _RecordingPipeline.init_kwargs
        assert kwargs is not None
        # 共通 kwargs
        assert kwargs["device"] == "cpu"
        assert kwargs["threshold"] == 0.5
        assert kwargs["letterbox"] is True
        assert kwargs["pipeline_mode"] == "cpu"
        assert kwargs["use_fp16"] is False
        assert kwargs["phased_timer"].phases == _RecordingPipeline.PHASES
        # PipelineContext が組み上がっている
        assert isinstance(ctx, PipelineContext)
        assert ctx.precision == "fp32"
        assert ctx.actual_device == "cpu"

    def test_spec_kwargs_are_merged_into_pipeline(self, tmp_path: Path) -> None:
        """spec.build_pipeline_kwargs の戻り値が pipeline kwargs に merge される."""
        _RecordingPipeline.init_kwargs = None
        model_path = tmp_path / "model"
        model_path.mkdir()
        config = self._make_config(tmp_path)

        def extra(
            cfg: DetectionConfigDict,
            image_size: tuple[int, int],
            processor: Any | None,
        ) -> dict[str, Any]:
            assert processor == "PROCESSOR"
            return {
                "image_size": image_size,
                "nms_iou_threshold": cfg["nms_iou_threshold"],
                "processor": processor,
            }

        spec = ArchitectureSpec(
            pipeline_cls=_RecordingPipeline,
            backends=BackendFactories(
                pytorch=lambda p, d, fp16: _FakeBackend(),
                onnx=lambda p, d: _FakeBackend(),
                tensorrt=lambda p: _FakeBackend(),
                trt_available=False,
            ),
            load_processor=lambda mp, cfg: "PROCESSOR",
            build_pipeline_kwargs=extra,
        )

        setup_pipeline(spec, config, model_path)

        kwargs = _RecordingPipeline.init_kwargs
        assert kwargs is not None
        assert kwargs["image_size"] == (320, 320)
        assert kwargs["nms_iou_threshold"] == pytest.approx(0.4)
        assert kwargs["processor"] == "PROCESSOR"

    def test_default_image_size_used_when_config_missing(self, tmp_path: Path) -> None:
        """config に image_size が無い場合 spec.default_image_size が採用される."""
        _RecordingPipeline.init_kwargs = None
        model_path = tmp_path / "model"
        model_path.mkdir()
        config = self._make_config(tmp_path)
        del config["image_size"]

        captured: dict[str, tuple[int, int]] = {}

        def extra(
            cfg: DetectionConfigDict,
            image_size: tuple[int, int],
            processor: Any | None,
        ) -> dict[str, Any]:
            captured["image_size"] = image_size
            return {}

        spec = ArchitectureSpec(
            pipeline_cls=_RecordingPipeline,
            backends=BackendFactories(
                pytorch=lambda p, d, fp16: _FakeBackend(),
                onnx=lambda p, d: _FakeBackend(),
                tensorrt=lambda p: _FakeBackend(),
                trt_available=False,
            ),
            build_pipeline_kwargs=extra,
            default_image_size=(256, 512),
        )

        setup_pipeline(spec, config, model_path)

        assert captured["image_size"] == (256, 512)
