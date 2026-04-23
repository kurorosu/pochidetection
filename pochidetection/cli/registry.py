"""アーキテクチャごとの train / infer / build_pipeline を解決するレジストリ."""

from collections.abc import Callable
from typing import Protocol

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.core.types import BuildPipelineFn

__all__ = [
    "InferFn",
    "TrainFn",
    "get_build_pipeline_for_arch",
    "get_infer_for_arch",
    "resolve_train",
]

TrainFn = Callable[[DetectionConfigDict, str], None]


class InferFn(Protocol):
    """画像推論関数のプロトコル."""

    def __call__(
        self,
        config: DetectionConfigDict,
        image_dir: str,
        model_dir: str | None = None,
        config_path: str | None = None,
        *,
        save_crop: bool = True,
    ) -> None:
        """画像推論を実行する."""
        ...


# Why: アーキテクチャ別モジュールの遅延ロード. 未使用アーキテクチャの重い依存
# (transformers, torchvision models 等) を CLI 起動時に読み込まないため.
def _import_rtdetr_train() -> TrainFn:
    from pochidetection.scripts.rtdetr.train import train

    return train


def _import_rtdetr_infer() -> InferFn:
    from pochidetection.scripts.rtdetr.infer import infer

    return infer


def _import_rtdetr_build_pipeline() -> BuildPipelineFn:
    from pochidetection.scripts.rtdetr.infer import build_pipeline

    return build_pipeline


def _import_ssdlite_train() -> TrainFn:
    from pochidetection.scripts.ssdlite.train import train

    return train


def _import_ssdlite_infer() -> InferFn:
    from pochidetection.scripts.ssdlite.infer import infer

    return infer


def _import_ssdlite_build_pipeline() -> BuildPipelineFn:
    from pochidetection.scripts.ssdlite.infer import build_pipeline

    return build_pipeline


def _import_ssd300_train() -> TrainFn:
    from pochidetection.scripts.ssd300.train import train

    return train


def _import_ssd300_infer() -> InferFn:
    from pochidetection.scripts.ssd300.infer import infer

    return infer


def _import_ssd300_build_pipeline() -> BuildPipelineFn:
    from pochidetection.scripts.ssd300.infer import build_pipeline

    return build_pipeline


_REGISTRY: dict[
    str,
    dict[
        str,
        Callable[[], TrainFn] | Callable[[], InferFn] | Callable[[], BuildPipelineFn],
    ],
] = {
    "RTDetr": {
        "train": _import_rtdetr_train,
        "infer": _import_rtdetr_infer,
        "build_pipeline": _import_rtdetr_build_pipeline,
    },
    "SSDLite": {
        "train": _import_ssdlite_train,
        "infer": _import_ssdlite_infer,
        "build_pipeline": _import_ssdlite_build_pipeline,
    },
    "SSD300": {
        "train": _import_ssd300_train,
        "infer": _import_ssd300_infer,
        "build_pipeline": _import_ssd300_build_pipeline,
    },
}

_DEFAULT_ARCH = "RTDetr"


def resolve_train(config: DetectionConfigDict) -> TrainFn:
    """Architecture に基づいて train 関数を返す.

    Args:
        config: 設定辞書.

    Returns:
        train 関数.
    """
    arch = config.get("architecture", _DEFAULT_ARCH)
    loader = _REGISTRY[arch]["train"]
    return loader()  # type: ignore[return-value]


def get_infer_for_arch(config: DetectionConfigDict) -> InferFn:
    """Architecture に基づいて infer 関数を返す.

    Args:
        config: 設定辞書.

    Returns:
        infer 関数.
    """
    arch = config.get("architecture", _DEFAULT_ARCH)
    loader = _REGISTRY[arch]["infer"]
    return loader()  # type: ignore[return-value]


def get_build_pipeline_for_arch(config: DetectionConfigDict) -> BuildPipelineFn:
    """Architecture に基づいて build_pipeline 関数を返す.

    Args:
        config: 設定辞書.

    Returns:
        build_pipeline 関数.
    """
    arch = config.get("architecture", _DEFAULT_ARCH)
    loader = _REGISTRY[arch]["build_pipeline"]
    return loader()  # type: ignore[return-value]
