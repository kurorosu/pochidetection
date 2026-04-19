"""RTDetr / SSD 共通: `pipeline_mode` プロパティのテスト.

RTDetr / SSD の pipeline_mode 挙動は共通仕様のため, parametrize で統合する.
API 差分 (RTDetr は processor / image_size 必須, SSD は image_size 必須) は
factory 関数で吸収する.
"""

from typing import Any, Callable, Literal

import pytest
import torch
from torchvision.transforms import v2

from pochidetection.interfaces import IInferenceBackend
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.pipelines.rtdetr_pipeline import RTDetrPipeline
from pochidetection.pipelines.ssd_pipeline import SsdPipeline

_DUMMY_IMAGE_SIZE = (64, 64)


class _DummyRTDetrBackend(IInferenceBackend[tuple[torch.Tensor, torch.Tensor]]):
    """RTDetr 用のダミー backend."""

    def infer(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ダミー推論."""
        return torch.zeros((1, 100, 2)), torch.zeros((1, 100, 4))


class _DummySsdBackend(IInferenceBackend[dict[str, torch.Tensor]]):
    """SSD 用のダミー backend."""

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """ダミー推論."""
        return {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }


class _DummyProcessor:
    """RTDetr 用のダミー processor."""

    def post_process_object_detection(
        self, outputs: Any, target_sizes: Any, threshold: float
    ) -> list[dict[str, Any]]:
        """ダミー後処理."""
        return [
            {
                "scores": torch.tensor([]),
                "labels": torch.tensor([], dtype=torch.int64),
                "boxes": torch.zeros((0, 4)),
            }
        ]


def _make_transform() -> v2.Compose:
    """テスト用 transform を生成."""
    return v2.Compose(
        [
            v2.Resize(_DUMMY_IMAGE_SIZE, interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def _make_rtdetr(
    pipeline_mode: Literal["cpu", "gpu"] | None = None,
) -> RTDetrPipeline:
    """RTDetr pipeline factory. pipeline_mode=None はデフォルト引数を使用."""
    kwargs: dict[str, Any] = {
        "backend": _DummyRTDetrBackend(),
        "processor": _DummyProcessor(),
        "transform": _make_transform(),
        "device": "cpu",
    }
    if pipeline_mode is not None:
        kwargs["pipeline_mode"] = pipeline_mode
    if pipeline_mode == "gpu":
        kwargs["image_size"] = _DUMMY_IMAGE_SIZE
    return RTDetrPipeline(**kwargs)


def _make_ssd(
    pipeline_mode: Literal["cpu", "gpu"] | None = None,
) -> SsdPipeline:
    """SSD pipeline factory. pipeline_mode=None はデフォルト引数を使用."""
    kwargs: dict[str, Any] = {
        "backend": _DummySsdBackend(),
        "transform": _make_transform(),
        "image_size": _DUMMY_IMAGE_SIZE,
        "device": "cpu",
    }
    if pipeline_mode is not None:
        kwargs["pipeline_mode"] = pipeline_mode
    return SsdPipeline(**kwargs)


_PIPELINE_FACTORIES: list[
    tuple[str, Callable[[Literal["cpu", "gpu"] | None], IDetectionPipeline]]
] = [
    ("rtdetr", _make_rtdetr),
    ("ssd", _make_ssd),
]


class TestPipelineModeProperty:
    """pipeline_mode プロパティの挙動を両 Pipeline で共通検証する."""

    @pytest.mark.parametrize(
        ("name", "factory"),
        _PIPELINE_FACTORIES,
        ids=[n for n, _ in _PIPELINE_FACTORIES],
    )
    def test_default_is_cpu(
        self,
        name: str,
        factory: Callable[[Literal["cpu", "gpu"] | None], IDetectionPipeline],
    ) -> None:
        """pipeline_mode 未指定時は 'cpu' (後方互換)."""
        pipeline = factory(None)
        assert pipeline.pipeline_mode == "cpu"

    @pytest.mark.parametrize(
        ("name", "factory"),
        _PIPELINE_FACTORIES,
        ids=[n for n, _ in _PIPELINE_FACTORIES],
    )
    @pytest.mark.parametrize("mode", ["cpu", "gpu"])
    def test_explicit_mode_is_reflected(
        self,
        name: str,
        factory: Callable[[Literal["cpu", "gpu"] | None], IDetectionPipeline],
        mode: Literal["cpu", "gpu"],
    ) -> None:
        """明示した pipeline_mode がプロパティに反映される."""
        pipeline = factory(mode)
        assert pipeline.pipeline_mode == mode
