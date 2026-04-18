"""IDetectionBackend.predict の phase_times 構成を classical test で検証."""

from pathlib import Path
from typing import Any

import numpy as np

from pochidetection.api.backends import PyTorchDetectionBackend
from pochidetection.core.detection import Detection
from pochidetection.interfaces.pipeline import IDetectionPipeline, ImageInput


class _StubPipeline(IDetectionPipeline[Any, Any]):
    """run() の戻り値を固定し, GPU event 計測値も設定可能なスタブ."""

    def __init__(self, *, gpu_inference_ms: float | None = None) -> None:
        self._validate_phased_timer(None)
        self._last_inference_gpu_ms = gpu_inference_ms
        self.run_calls: list[np.ndarray] = []

    def run(self, image: ImageInput) -> list[Detection]:
        """画像を記録し空の検出リストを返す."""
        assert isinstance(image, np.ndarray)
        self.run_calls.append(image.copy())
        return []


def _make_backend(pipeline: _StubPipeline) -> PyTorchDetectionBackend:
    return PyTorchDetectionBackend(
        pipeline=pipeline,
        config={
            "architecture": "stub",
            "num_classes": 1,
            "class_names": ["dummy"],
            "image_size": {"height": 32, "width": 32},
        },
        model_path=Path("dummy.pt"),
    )


def test_predict_omits_gpu_ms_key_when_pipeline_reports_none() -> None:
    """CPU 等で pipeline.last_inference_gpu_ms が None なら phase_times に
    pipeline_inference_gpu_ms キーは含まれない.
    """
    pipeline = _StubPipeline(gpu_inference_ms=None)
    backend = _make_backend(pipeline)
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    _, phase_times = backend.predict(image)

    assert "pipeline_inference_gpu_ms" not in phase_times
    # phase_times は 4 値 (CUDA 時) / 3 値 (CPU 時) のみで,
    # cvt_color_ms / pipeline_total_ms 等の breakdown を含まない.
    assert "cvt_color_ms" not in phase_times
    assert "pipeline_total_ms" not in phase_times


def test_predict_includes_gpu_ms_when_pipeline_reports_value() -> None:
    """pipeline.last_inference_gpu_ms が set されていれば phase_times に転記される."""
    pipeline = _StubPipeline(gpu_inference_ms=7.42)
    backend = _make_backend(pipeline)
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    _, phase_times = backend.predict(image)

    assert phase_times["pipeline_inference_gpu_ms"] == 7.42
