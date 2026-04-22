"""推論 pipeline 構築のアーキテクチャ固有情報 (``ArchitectureSpec``) と共通 setup.

アーキ固有部分 (pipeline クラス / backend factory / processor / transform /
kwargs 組立) を ``ArchitectureSpec`` dataclass で束ね, ``setup_pipeline`` に
共通の手順 (cudnn 設定 / backend 生成 / device 解決 / pipeline_mode 解決 /
PhasedTimer 生成 / pipeline 構築 / context 構築) を集約する.

``resolve_and_setup_pipeline`` はモデル解決まで一段階前に戻り, CLI / WebAPI /
video / stream の全経路からモデルパス解決 + pipeline 構築を 1 コールで済ませる
ためのエントリポイント.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torchvision.transforms import v2

from pochidetection.cli.registry import resolve_setup_pipeline
from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.core.coco_classes import PRETRAINED_CONFIG_PATH
from pochidetection.core.types import SetupPipelineFn
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.logging import LoggerManager
from pochidetection.pipelines.backend_factory import create_backend
from pochidetection.pipelines.context import (
    PipelineContext,
    ResolvedPipeline,
    build_pipeline_context,
)
from pochidetection.pipelines.model_path import PRETRAINED, _resolve_model_path
from pochidetection.pipelines.runtime import (
    resolve_device,
    resolve_pipeline_mode,
    setup_cudnn_benchmark,
)
from pochidetection.utils import PhasedTimer
from pochidetection.utils.config_loader import ConfigLoader

__all__ = [
    "ArchitectureSpec",
    "BackendFactories",
    "resolve_and_setup_pipeline",
    "setup_pipeline",
]

logger = LoggerManager().get_logger(__name__)


_BackendFactoryPytorch = Callable[[Path, str, bool], IInferenceBackend[Any]]
_BackendFactoryOnnx = Callable[[Path, str], IInferenceBackend[Any]]
_BackendFactoryTrt = Callable[[Path], IInferenceBackend[Any]]

# pipeline クラスへ渡す追加 kwargs を組み立てる関数.
# 引数: (config, image_size_hw, processor_or_none) → dict of kwargs.
_PipelineKwargsFn = Callable[
    [DetectionConfigDict, tuple[int, int], Any | None], dict[str, Any]
]

# target_hw (height, width) から v2.Compose を組み立てる関数.
_TransformBuilder = Callable[[tuple[int, int]], v2.Compose]

# model_path / config から processor (RT-DETR の RTDetrImageProcessor 等) を
# ロードする関数. processor 不要な SSD 系では ``None`` を指定する.
_ProcessorLoader = Callable[[Path, DetectionConfigDict], Any]


@dataclass(frozen=True, slots=True)
class BackendFactories:
    """推論 backend を 3 種類 (PyTorch / ONNX / TensorRT) 生成する factory の束.

    Attributes:
        pytorch: ``(model_path, device, use_fp16)`` を受け取り PyTorch backend を
            返す. FP16 適用 (model.half()) は factory 側で行う.
        onnx: ``(model_path, device)`` を受け取り ONNX backend を返す.
        tensorrt: ``(model_path,)`` を受け取り TensorRT backend を返す.
            ``trt_available=False`` の場合は呼ばれないので未対応アーキでは
            ``NotImplementedError`` を送出するダミー関数を渡せば良い.
        trt_available: 実装環境で TensorRT が利用可能かどうか.
    """

    pytorch: _BackendFactoryPytorch
    onnx: _BackendFactoryOnnx
    tensorrt: _BackendFactoryTrt
    trt_available: bool = False


def _default_transform_builder(image_size: tuple[int, int]) -> v2.Compose:
    """既定の v2.Compose (Resize → ToImage → ToDtype float32/scale)."""
    return v2.Compose(
        [
            v2.Resize(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def _empty_pipeline_kwargs(
    config: DetectionConfigDict,
    image_size: tuple[int, int],
    processor: Any | None,
) -> dict[str, Any]:
    """追加 kwargs が不要なアーキ向けのデフォルト (空辞書)."""
    del config, image_size, processor  # unused
    return {}


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    """推論 pipeline 構築のアーキテクチャ固有情報.

    ``setup_pipeline`` と組み合わせて使う. アーキ固有部分 (pipeline クラス /
    backend factory / processor / transform / kwargs 組立) を dataclass で束ね,
    共通の setup 手順は ``setup_pipeline`` 側に集約する.

    Attributes:
        pipeline_cls: 構築する Pipeline クラス (``IDetectionPipeline`` 派生).
            ``PHASES`` クラス変数から PhasedTimer の phase を解決する.
        backends: 3 種 backend の factory 束.
        load_processor: processor (HF ImageProcessor 等) をロードする関数.
            processor 不要なアーキでは ``None``.
        build_transform: ``image_size`` から前処理 ``v2.Compose`` を組み立てる関数.
        build_pipeline_kwargs: pipeline クラスへ渡すアーキ固有 kwargs を返す関数.
            共通 kwargs (``backend`` / ``transform`` / ``threshold`` 等) は
            ``setup_pipeline`` 側で付与するため, ここでは差分のみ返す.
        default_image_size: ``config["image_size"]`` 未指定時のフォールバック
            ``(height, width)``.
    """

    pipeline_cls: type[IDetectionPipeline[Any, Any]]
    backends: BackendFactories
    load_processor: _ProcessorLoader | None = None
    build_transform: _TransformBuilder = field(default=_default_transform_builder)
    build_pipeline_kwargs: _PipelineKwargsFn = field(default=_empty_pipeline_kwargs)
    default_image_size: tuple[int, int] = (640, 640)


def setup_pipeline(
    spec: ArchitectureSpec,
    config: DetectionConfigDict,
    model_path: Path,
) -> PipelineContext:
    """アーキテクチャ固有 spec と config から PipelineContext を構築する.

    3 アーキ (RT-DETR / SSDLite / SSD300) 共通の boilerplate
    (cudnn 設定 / backend 生成 / device 解決 / pipeline_mode 解決 /
    PhasedTimer 生成 / pipeline 構築 / context 構築) を集約する.

    Args:
        spec: アーキテクチャ固有情報.
        config: 検証済み設定辞書.
        model_path: モデルのパス.

    Returns:
        構築済みの ``PipelineContext``.
    """
    setup_cudnn_benchmark(config)

    processor = (
        spec.load_processor(model_path, config)
        if spec.load_processor is not None
        else None
    )

    backend, precision, _ = create_backend(
        model_path,
        config,
        create_trt=spec.backends.tensorrt,
        create_onnx=spec.backends.onnx,
        create_pytorch=spec.backends.pytorch,
        trt_available=spec.backends.trt_available,
    )

    image_size_cfg = config.get(
        "image_size",
        {"height": spec.default_image_size[0], "width": spec.default_image_size[1]},
    )
    image_size = (int(image_size_cfg["height"]), int(image_size_cfg["width"]))

    actual_device, runtime_device = resolve_device(model_path, config, backend)
    pipeline_mode = resolve_pipeline_mode(config.get("pipeline_mode"), model_path)

    transform = spec.build_transform(image_size)
    phased_timer = PhasedTimer(phases=spec.pipeline_cls.PHASES, device=runtime_device)

    # precision は "fp16" / "fp32" で, pipeline には bool で渡す.
    use_fp16 = precision == "fp16"

    pipeline_kwargs: dict[str, Any] = {
        "backend": backend,
        "transform": transform,
        "device": runtime_device,
        "threshold": config["infer_score_threshold"],
        "use_fp16": use_fp16,
        "phased_timer": phased_timer,
        "pipeline_mode": pipeline_mode,
        "letterbox": config.get("letterbox", True),
    }
    pipeline_kwargs.update(spec.build_pipeline_kwargs(config, image_size, processor))

    pipeline = spec.pipeline_cls(**pipeline_kwargs)

    return build_pipeline_context(
        pipeline=pipeline,
        phased_timer=phased_timer,
        config=config,
        model_path=model_path,
        actual_device=actual_device,
        precision=precision,
    )


def resolve_and_setup_pipeline(
    config: DetectionConfigDict,
    model_dir: str | None,
    config_path: str | None,
    logger_instance: logging.Logger | None = None,
) -> ResolvedPipeline | None:
    """モデルパスを解決し, パイプラインを構築する.

    model_dir が None の場合はプリトレインモデルにフォールバックする.
    model_dir 指定時にモデルが見つからない場合は None を返す.

    Args:
        config: 設定辞書.
        model_dir: モデルディレクトリ (None でプリトレイン).
        config_path: 設定ファイルのパス.
        logger_instance: ロガー. None の場合はモジュールロガーを使用.

    Returns:
        解決済みパイプライン情報. モデル未発見時は None.
    """
    log = logger_instance or logger

    if model_dir is not None:
        model_path = _resolve_model_path(config, model_dir)
        if model_path is None:
            return None
    else:
        model_path = PRETRAINED

    if model_path == PRETRAINED:
        config_path = PRETRAINED_CONFIG_PATH
        config = ConfigLoader.load(PRETRAINED_CONFIG_PATH)
        setup_pipeline_fn: SetupPipelineFn = resolve_setup_pipeline(config)
        log.info("Loading RT-DETR COCO pretrained model")
    else:
        setup_pipeline_fn = resolve_setup_pipeline(config)
        log.info(f"Loading model from {model_path}")

    ctx = setup_pipeline_fn(config, model_path)
    return ResolvedPipeline(
        ctx=ctx, config=config, config_path=config_path, model_path=model_path
    )
