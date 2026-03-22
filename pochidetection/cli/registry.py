"""アーキテクチャごとの train / infer / setup_pipeline を解決するレジストリ."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from pochidetection.configs.schemas import DetectionConfigDict

TrainFn = Callable[[DetectionConfigDict, str], None]
SetupPipelineFn = Callable[[DetectionConfigDict, Path], Any]


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


def _import_rtdetr_train() -> TrainFn:
    from pochidetection.scripts.rtdetr.train import train

    return train


def _import_rtdetr_infer() -> InferFn:
    from pochidetection.scripts.rtdetr.infer import infer

    return infer


def _import_rtdetr_setup_pipeline() -> SetupPipelineFn:
    from pochidetection.scripts.rtdetr.infer import _setup_pipeline

    return _setup_pipeline


def _import_ssdlite_train() -> TrainFn:
    from pochidetection.scripts.ssdlite.train import train

    return train


def _import_ssdlite_infer() -> InferFn:
    from pochidetection.scripts.ssdlite.infer import infer

    return infer


def _import_ssdlite_setup_pipeline() -> SetupPipelineFn:
    from pochidetection.scripts.ssdlite.infer import _setup_pipeline

    return _setup_pipeline


def _import_ssd300_train() -> TrainFn:
    from pochidetection.scripts.ssd300.train import train

    return train


def _import_ssd300_infer() -> InferFn:
    from pochidetection.scripts.ssd300.infer import infer

    return infer


def _import_ssd300_setup_pipeline() -> SetupPipelineFn:
    from pochidetection.scripts.ssd300.infer import _setup_pipeline

    return _setup_pipeline


_REGISTRY: dict[
    str,
    dict[
        str,
        Callable[[], TrainFn] | Callable[[], InferFn] | Callable[[], SetupPipelineFn],
    ],
] = {
    "RTDetr": {
        "train": _import_rtdetr_train,
        "infer": _import_rtdetr_infer,
        "setup_pipeline": _import_rtdetr_setup_pipeline,
    },
    "SSDLite": {
        "train": _import_ssdlite_train,
        "infer": _import_ssdlite_infer,
        "setup_pipeline": _import_ssdlite_setup_pipeline,
    },
    "SSD300": {
        "train": _import_ssd300_train,
        "infer": _import_ssd300_infer,
        "setup_pipeline": _import_ssd300_setup_pipeline,
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


def resolve_infer(config: DetectionConfigDict) -> InferFn:
    """Architecture に基づいて infer 関数を返す.

    Args:
        config: 設定辞書.

    Returns:
        infer 関数.
    """
    arch = config.get("architecture", _DEFAULT_ARCH)
    loader = _REGISTRY[arch]["infer"]
    return loader()  # type: ignore[return-value]


def resolve_setup_pipeline(config: DetectionConfigDict) -> SetupPipelineFn:
    """Architecture に基づいて _setup_pipeline 関数を返す.

    Args:
        config: 設定辞書.

    Returns:
        setup_pipeline 関数.
    """
    arch = config.get("architecture", _DEFAULT_ARCH)
    loader = _REGISTRY[arch]["setup_pipeline"]
    return loader()  # type: ignore[return-value]
