"""pipelines/spec.py (ArchitectureSpec + build_pipeline_from_spec) のテスト."""

from pathlib import Path
from typing import Any

import pytest

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.core.detection import Detection
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.interfaces.pipeline import IDetectionPipeline, ImageInput
from pochidetection.pipelines.context import PipelineContext
from pochidetection.pipelines.spec import (
    ArchitectureSpec,
    BackendFactories,
    build_pipeline_from_spec,
)


class _FakeBackend(IInferenceBackend[Any]):
    """最小 stub backend."""

    def infer(self, inputs: Any) -> Any:
        """空結果を返す."""
        return None


class _RecordingPipeline(IDetectionPipeline[Any, Any]):
    """build_pipeline_from_spec が渡した kwargs を ``init_kwargs`` に記録する stub pipeline.

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
    """ArchitectureSpec + build_pipeline_from_spec の統合テスト."""

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

        ctx = build_pipeline_from_spec(spec, config, model_path)

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

        build_pipeline_from_spec(spec, config, model_path)

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

        build_pipeline_from_spec(spec, config, model_path)

        assert captured["image_size"] == (256, 512)
