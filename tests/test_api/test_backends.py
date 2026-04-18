"""IDetectionBackend.warmup の振る舞いを classical test で検証."""

from pathlib import Path
from typing import Any

import numpy as np

from pochidetection.api.backends import _WARMUP_ITERATIONS, PyTorchDetectionBackend
from pochidetection.core.detection import Detection
from pochidetection.interfaces.pipeline import IDetectionPipeline, ImageInput


class _StubPipeline(IDetectionPipeline[Any, Any]):
    """run() が呼ばれた回数と渡された画像を記録するだけのスタブ."""

    def __init__(self) -> None:
        self._validate_phased_timer(None)
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


def test_warmup_runs_three_iterations() -> None:
    """warmup() は _WARMUP_ITERATIONS 回 pipeline.run を呼ぶ."""
    pipeline = _StubPipeline()
    backend = _make_backend(pipeline)

    backend.warmup()

    assert len(pipeline.run_calls) == _WARMUP_ITERATIONS == 3


def test_warmup_uses_random_uint8_images_with_config_size() -> None:
    """ダミー画像は config の image_size と uint8 で生成される."""
    pipeline = _StubPipeline()
    backend = _make_backend(pipeline)

    backend.warmup()

    for image in pipeline.run_calls:
        assert image.shape == (32, 32, 3)
        assert image.dtype == np.uint8
    # zeros ではなく乱数なので少なくとも 1 件は非ゼロを含む.
    assert any(image.any() for image in pipeline.run_calls)
