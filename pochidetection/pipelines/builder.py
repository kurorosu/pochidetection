"""推論の共通ロジック.

RT-DETR と SSDLite で共有される推論エントリ, レポート出力,
ベンチマークサマリーのロジックを提供する.

"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

import torch
from PIL import Image
from torchvision.transforms import v2

from pochidetection.cli.registry import resolve_setup_pipeline
from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.core.coco_classes import PRETRAINED_CONFIG_PATH
from pochidetection.core.detection import Detection
from pochidetection.core.types import SetupPipelineFn
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.logging import LoggerManager
from pochidetection.reporting import (
    DetectionSummary,
    InferenceSaver,
    Visualizer,
    build_detection_results,
    build_detection_summary,
    write_detection_results_csv,
    write_detection_summary,
)
from pochidetection.utils import (
    BenchmarkResult,
    DetectionMetrics,
    PhasedTimer,
    WorkspaceManager,
    build_benchmark_result,
    write_benchmark_result,
)
from pochidetection.utils.config_loader import ConfigLoader
from pochidetection.utils.device import is_fp16_available
from pochidetection.utils.infer_debug import InferDebugConfig, save_infer_debug_image
from pochidetection.utils.map_evaluator import MapEvaluator
from pochidetection.visualization import (
    ConfusionMatrixPlotter,
    LabelMapper,
    build_confusion_matrix,
)

logger = LoggerManager().get_logger(__name__)

__all__ = [
    "PRETRAINED",
    "ArchitectureSpec",
    "BackendFactories",
    "PipelineContext",
    "ResolvedPipeline",
    "build_pipeline_context",
    "create_backend",
    "infer",
    "is_onnx_model",
    "is_tensorrt_model",
    "resolve_and_setup_pipeline",
    "resolve_device",
    "resolve_pipeline_mode",
    "setup_cudnn_benchmark",
    "setup_pipeline",
]


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

PRETRAINED = Path("__pretrained__")
"""プリトレインモデル使用を示すセンチネル値."""

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


# ---------------------------------------------------------------------------
# データ構造 (NamedTuple / Protocol)
# ---------------------------------------------------------------------------


class PipelineContext(NamedTuple):
    """推論パイプラインコンテキスト."""

    pipeline: IDetectionPipeline
    phased_timer: PhasedTimer
    visualizer: Visualizer
    saver: InferenceSaver
    label_mapper: LabelMapper | None
    class_names: list[str] | None
    actual_device: str
    precision: str


class ResolvedPipeline(NamedTuple):
    """モデル解決・パイプライン構築の結果."""

    ctx: PipelineContext
    config: DetectionConfigDict
    config_path: str | None
    model_path: Path


class _InferenceContext(Protocol):
    """推論コンテキストのプロトコル (内部利用)."""

    @property
    def pipeline(self) -> IDetectionPipeline:
        """推論パイプライン."""
        ...

    @property
    def phased_timer(self) -> PhasedTimer:
        """PhasedTimer."""
        ...

    @property
    def saver(self) -> InferenceSaver:
        """InferenceSaver."""
        ...

    @property
    def label_mapper(self) -> LabelMapper | None:
        """LabelMapper."""
        ...

    @property
    def class_names(self) -> list[str] | None:
        """クラス名リスト."""
        ...

    @property
    def actual_device(self) -> str:
        """実際のデバイス名."""
        ...

    @property
    def precision(self) -> str:
        """精度 (fp32/fp16)."""
        ...

    @property
    def visualizer(self) -> Visualizer:
        """Visualizer."""
        ...


# ---------------------------------------------------------------------------
# モデルパス解決
# ---------------------------------------------------------------------------


def is_onnx_model(model_path: Path) -> bool:
    """モデルパスが ONNX ファイルかどうかを判定する.

    Args:
        model_path: モデルのパス.

    Returns:
        .onnx ファイルの場合 True.
    """
    return model_path.suffix.lower() == ".onnx"


def is_tensorrt_model(model_path: Path) -> bool:
    """モデルパスが TensorRT エンジンかどうかを判定する.

    Args:
        model_path: モデルのパス.

    Returns:
        .engine ファイルの場合 True.
    """
    return model_path.suffix.lower() == ".engine"


def _resolve_model_path(
    config: DetectionConfigDict,
    model_dir: str | None,
) -> Path | None:
    """モデルパスを解決.

    Args:
        config: 設定辞書.
        model_dir: 指定されたモデルディレクトリ.

    Returns:
        モデルパス. エラー時は None.
    """
    if model_dir is not None:
        model_path = Path(model_dir)
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return None
        return model_path

    workspace_manager = WorkspaceManager(config["work_dir"])
    workspaces = workspace_manager.get_available_workspaces()

    if not workspaces:
        logger.info(
            "No trained models found. Using COCO pretrained model for inference."
        )
        return PRETRAINED

    latest_workspace = Path(str(workspaces[-1]["path"]))
    model_path = latest_workspace / "best"

    if not model_path.exists():
        logger.error(
            f"Best model not found at {model_path}. Please run training first."
        )
        return None

    return model_path


# ---------------------------------------------------------------------------
# Backend / Pipeline 構築
# ---------------------------------------------------------------------------

# バックエンドファクトリコールバックの型
_CreateTrtFn = Callable[[Path], IInferenceBackend[Any]]
_CreateOnnxFn = Callable[[Path, str], IInferenceBackend[Any]]
_CreatePytorchFn = Callable[[Path, str, bool], IInferenceBackend[Any]]


def resolve_pipeline_mode(
    requested: Literal["cpu", "gpu"] | None,
    model_path: Path,
) -> Literal["cpu", "gpu"]:
    """Preprocess の経路を backend 種別から解決する.

    PyTorch / TensorRT は default 'gpu', ONNX は default 'cpu'.
    ONNX で 'gpu' を明示指定された場合は ValueError で起動を拒否する
    (ONNX Runtime は CPU numpy を要求するため GPU preprocess の効果がないため).

    Args:
        requested: CLI / config で指定された値. None の場合は backend 種別から決定.
        model_path: モデルのパス (backend 種別判定用).

    Returns:
        解決後の経路名 ('cpu' or 'gpu').

    Raises:
        ValueError: ONNX backend で 'gpu' を明示指定された場合.
    """
    if is_onnx_model(model_path):
        if requested == "gpu":
            raise ValueError(
                "ONNX backend は --pipeline cpu のみ対応. "
                "フォールバック手順: "
                "(1) `--pipeline cpu` を指定するか未指定にする, "
                "または (2) GPU preprocess を使う場合は PyTorch (.pth) / "
                "TensorRT (.engine) バックエンドのモデルに切り替える."
            )
        return "cpu"
    return requested if requested is not None else "gpu"


def create_backend(
    model_path: Path,
    config: DetectionConfigDict,
    create_trt: _CreateTrtFn,
    create_onnx: _CreateOnnxFn,
    create_pytorch: _CreatePytorchFn,
    trt_available: bool = False,
) -> tuple[IInferenceBackend[Any], str, bool]:
    """モデルパスからバックエンドを生成する.

    TensorRT / ONNX / PyTorch の分岐ロジックを共通化し,
    具象バックエンドの生成はコールバックに委譲する.

    Args:
        model_path: モデルのパス.
        config: 設定辞書.
        create_trt: TensorRT バックエンド生成コールバック.
            (model_path,) を受け取り IInferenceBackend を返す.
        create_onnx: ONNX バックエンド生成コールバック.
            (model_path, device) を受け取り IInferenceBackend を返す.
        create_pytorch: PyTorch バックエンド生成コールバック.
            (model_path, device, use_fp16) を受け取り IInferenceBackend を返す.
            FP16 適用 (model.half()) はコールバック側で行う.
        trt_available: TensorRT が利用可能かどうか.

    Returns:
        (backend, precision, use_fp16) のタプル.
    """
    device = config["device"]
    use_fp16 = config.get("use_fp16", False)

    if is_tensorrt_model(model_path):
        if not trt_available:
            raise ImportError(
                "tensorrt パッケージがインストールされていません. "
                "TensorRT バックエンドを使用するには TensorRT をインストールしてください."
            )
        logger.info("TensorRT backend selected")
        return create_trt(model_path), "fp32", False

    if is_onnx_model(model_path):
        logger.info("ONNX backend selected")
        return create_onnx(model_path, device), "fp32", False

    fp16 = is_fp16_available(use_fp16, device)
    backend = create_pytorch(model_path, device, fp16)

    if fp16:
        logger.info("FP16 enabled")

    precision = "fp16" if fp16 else "fp32"
    return backend, precision, use_fp16


def setup_cudnn_benchmark(config: DetectionConfigDict) -> None:
    """cudnn.benchmark を設定する.

    Args:
        config: 設定辞書.
    """
    device = config["device"]
    if config.get("cudnn_benchmark", False) and device == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cudnn.benchmark enabled")


def resolve_device(
    model_path: Path,
    config: DetectionConfigDict,
    backend: IInferenceBackend[Any],
) -> tuple[str, str]:
    """モデル形式に応じたデバイスを解決する.

    backend 種別ごとに以下のように決定する:

    - TensorRT (.engine): ``("cuda", "cuda")`` 固定. 推論も preprocess も GPU.
    - ONNX (.onnx): ``(actual_device, "cpu")``.
      ``actual_device`` は ``backend.active_providers`` に
      ``CUDAExecutionProvider`` が含まれるかで ``"cuda"`` / ``"cpu"`` を判定する
      (ログ表示・メトリクス用). 一方 ``runtime_device`` は常に ``"cpu"``
      とする. ONNX Runtime の ``session.run()`` は入力を CPU 上の numpy
      配列で受け取るため, preprocess 結果を GPU テンソルで渡しても
      Runtime 側で CPU へコピーし直され無駄が生じる. このため
      preprocess の配置先 (=``runtime_device``) を CPU に固定し,
      GPU preprocess 経路を選ばせないようにしている.
    - PyTorch (.pth): ``(device, device)``. config の ``device`` をそのまま使う.

    Args:
        model_path: モデルのパス.
        config: 設定辞書.
        backend: 生成済みのバックエンド. ONNX の場合のみ
            ``active_providers`` 属性を参照する.

    Returns:
        ``(actual_device, runtime_device)`` のタプル.

        - ``actual_device``: 推論が実際に走るデバイス (ログ / メトリクス用).
        - ``runtime_device``: preprocess 結果の配置先デバイス.
          pipeline builder がこの値に従って CPU / GPU preprocess 経路を切り替える.
    """
    device = config["device"]

    if is_tensorrt_model(model_path):
        return "cuda", "cuda"

    if is_onnx_model(model_path):
        active_providers = getattr(backend, "active_providers", [])
        actual_device = "cuda" if "CUDAExecutionProvider" in active_providers else "cpu"
        return actual_device, "cpu"

    return device, device


# ---------------------------------------------------------------------------
# ArchitectureSpec: アーキテクチャ共通 setup_pipeline
# ---------------------------------------------------------------------------


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


def build_pipeline_context(
    *,
    pipeline: IDetectionPipeline,
    phased_timer: PhasedTimer,
    config: DetectionConfigDict,
    model_path: Path,
    actual_device: str,
    precision: str,
) -> PipelineContext:
    """共通の初期化ステップから PipelineContext を構築する.

    LabelMapper, Visualizer, InferenceSaver の構築を共通化する.

    Args:
        pipeline: 構築済みの推論パイプライン.
        phased_timer: 構築済みの PhasedTimer.
        config: 設定辞書.
        model_path: モデルのパス.
        actual_device: 実際のデバイス名.
        precision: 精度 (fp32/fp16).

    Returns:
        構築済みの PipelineContext.
    """
    class_names = config.get("class_names")
    label_mapper = LabelMapper(class_names) if class_names else None
    visualizer = Visualizer(label_mapper=label_mapper)

    if model_path == PRETRAINED:
        saver_base = Path(config.get("work_dir", "work_dirs")) / "pretrained"
    elif is_onnx_model(model_path) or is_tensorrt_model(model_path):
        saver_base = model_path.parent
    else:
        saver_base = model_path
    saver = InferenceSaver(saver_base)

    return PipelineContext(
        pipeline=pipeline,
        phased_timer=phased_timer,
        visualizer=visualizer,
        saver=saver,
        label_mapper=label_mapper,
        class_names=class_names,
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


# ---------------------------------------------------------------------------
# 推論実行
# ---------------------------------------------------------------------------


def _collect_image_files(image_dir: str) -> list[Path] | None:
    """画像ファイルを収集.

    Args:
        image_dir: 画像ディレクトリパス.

    Returns:
        画像ファイルリスト. エラー時は None.
    """
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return None

    image_files = [
        f for f in image_dir_path.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS
    ]

    if not image_files:
        logger.warning(f"No image files found in {image_dir}")
        return None

    logger.info(f"Found {len(image_files)} images in {image_dir}")
    return image_files


def infer(
    config: DetectionConfigDict,
    image_dir: str,
    model_dir: str | None = None,
    config_path: str | None = None,
    *,
    save_crop: bool = True,
) -> None:
    """フォルダ内の画像を一括推論.

    Args:
        config: 設定辞書.
        image_dir: 推論対象の画像フォルダパス.
        model_dir: モデルディレクトリ. None の場合は最新ワークスペースの best を使用.
        config_path: 設定ファイルのパス. 指定時は推論結果ディレクトリにコピーする.
        save_crop: True の場合, 検出ボックスのクロップ画像を保存する.
    """
    resolved = resolve_and_setup_pipeline(config, model_dir, config_path)
    if resolved is None:
        return

    image_files = _collect_image_files(image_dir)
    if image_files is None:
        return

    ctx, config, config_path, model_path = resolved
    logger.info(f"Results will be saved to {ctx.saver.output_dir}")

    all_predictions = _run_inference(image_files, ctx, config, save_crop=save_crop)
    _write_reports(config, image_files, all_predictions, ctx, model_path, config_path)


def _run_inference(
    image_files: list[Path],
    ctx: _InferenceContext,
    config: DetectionConfigDict,
    *,
    save_crop: bool = True,
) -> dict[str, list[Detection]]:
    """画像ループで推論を実行.

    Args:
        image_files: 推論対象の画像ファイルリスト.
        ctx: パイプラインコンテキスト.
        config: 設定辞書. ``infer_debug_save_count`` / ``image_size`` /
            ``letterbox`` を debug 保存で参照する.
        save_crop: True の場合, 検出ボックスのクロップ画像を保存する.

    Returns:
        ファイル名をキー, 検出結果リストを値とする辞書.
    """
    all_predictions: dict[str, list[Detection]] = {}

    infer_debug = InferDebugConfig.from_config(config, ctx.saver.output_dir)

    for i, image_file in enumerate(image_files):
        with Image.open(image_file) as img:
            image = img.convert("RGB")

        if infer_debug is not None and i < infer_debug.save_count:
            save_infer_debug_image(
                source_image=image,
                target_hw=infer_debug.target_hw,
                letterbox=infer_debug.letterbox,
                save_path=infer_debug.output_dir / f"infer_{i:04d}.jpg",
            )

        detections = ctx.pipeline.run(image)
        all_predictions[image_file.name] = detections

        if save_crop:
            ctx.saver.save_crops(image, detections, image_file.name, ctx.label_mapper)

        result_image = ctx.visualizer.draw(image, detections, inplace=True)
        output_path = ctx.saver.save(result_image, image_file.name)

        inf_timer = ctx.phased_timer.get_timer("inference")
        logger.info(
            f"  {image_file.name} ({inf_timer.last_time_ms:.1f}ms) - "
            f"{len(detections)} objects -> {output_path.name}"
        )

    return all_predictions


# ---------------------------------------------------------------------------
# レポート出力
# ---------------------------------------------------------------------------


def _write_reports(
    config: DetectionConfigDict,
    image_files: list[Path],
    all_predictions: dict[str, list[Detection]],
    ctx: _InferenceContext,
    model_path: Path,
    config_path: str | None = None,
) -> None:
    """レポート出力 (mAP, summary, CSV, confusion matrix, benchmark).

    Args:
        config: 設定辞書.
        image_files: 推論対象の画像ファイルリスト.
        all_predictions: ファイル名をキー, 検出結果リストを値とする辞書.
        ctx: 推論コンテキスト.
        model_path: モデルのパス.
        config_path: 設定ファイルのパス. 指定時は推論結果ディレクトリにコピーする.
    """
    if config_path is not None:
        _save_config(config, config_path, ctx.saver.output_dir)

    detection_metrics = _evaluate_map(config, all_predictions)

    summary = build_detection_summary(all_predictions, ctx.label_mapper)
    summary_path = write_detection_summary(ctx.saver.output_dir, summary)
    logger.info(f"Detection summary saved to {summary_path}")
    _log_detection_summary(summary)

    # 推論結果 CSV 出力
    annotation_path_str = config.get("annotation_path")
    annotation_path = Path(annotation_path_str) if annotation_path_str else None
    if annotation_path is not None and not annotation_path.exists():
        logger.warning(f"Annotation file not found: {annotation_path}")
        annotation_path = None

    csv_rows = build_detection_results(
        predictions=all_predictions,
        label_mapper=ctx.label_mapper,
        annotation_path=annotation_path,
    )
    csv_path = write_detection_results_csv(ctx.saver.output_dir, csv_rows)
    logger.info(f"Detection results CSV saved to {csv_path}")

    # 混同行列出力
    if annotation_path is not None and ctx.class_names is not None:
        cm = build_confusion_matrix(
            predictions=all_predictions,
            annotation_path=annotation_path,
            class_names=ctx.class_names,
        )
        cm_plotter = ConfusionMatrixPlotter(cm, ctx.class_names)
        cm_path = ctx.saver.output_dir / "confusion_matrix.html"
        cm_plotter.plot(cm_path)
        logger.info(f"Confusion matrix saved to {cm_path}")

    result = build_benchmark_result(
        phased_timer=ctx.phased_timer,
        num_images=len(image_files),
        device=ctx.actual_device,
        precision=ctx.precision,
        model_path=str(model_path),
        detection_metrics=detection_metrics,
    )

    json_path = write_benchmark_result(ctx.saver.output_dir, result)
    logger.info(f"Benchmark result saved to {json_path}")

    _log_benchmark_summary(result)
    logger.info(f"Results saved to {ctx.saver.output_dir}")


def _save_config(
    config: DetectionConfigDict, config_path: str, output_dir: Path
) -> None:
    """マージ済み設定辞書を推論結果ディレクトリに保存する.

    Args:
        config: マージ済みの設定辞書.
        config_path: 設定ファイルのパス (ファイル名の取得に使用).
        output_dir: 推論結果の出力ディレクトリ.
    """
    dst = output_dir / Path(config_path).name
    ConfigLoader.write_config(config, dst)
    logger.info(f"Config saved to {dst}")


def _evaluate_map(
    config: DetectionConfigDict,
    predictions: dict[str, list[Detection]],
) -> DetectionMetrics | None:
    """Config に annotation_path が指定されていれば mAP を計算する.

    Args:
        config: 設定辞書.
        predictions: ファイル名をキー, 検出結果リストを値とする辞書.

    Returns:
        DetectionMetrics. annotation_path 未指定時は None.
    """
    annotation_path_str = config.get("annotation_path")
    if annotation_path_str is None:
        return None

    annotation_path = Path(annotation_path_str)
    if not annotation_path.exists():
        logger.warning(f"Annotation file not found: {annotation_path}")
        return None

    logger.info(f"Evaluating mAP with annotation: {annotation_path}")
    evaluator = MapEvaluator(annotation_path)
    return evaluator.evaluate(predictions)


def _log_benchmark_summary(result: BenchmarkResult) -> None:
    """ベンチマーク結果のサマリーをログ出力する.

    Args:
        result: ベンチマーク結果.
    """
    m = result.metrics
    s = result.samples
    logger.info(
        f"Inference completed: {s.num_samples} images "
        f"({s.warmup_samples} warmup skipped), "
        f"avg {m.avg_e2e_ms:.1f}ms/image (E2E), "
        f"throughput {m.throughput_e2e_ips:.1f} IPS (E2E), "
        f"{m.throughput_inference_ips:.1f} IPS (inference)"
    )
    for phase_name, phase in m.phases.items():
        logger.info(
            f"  {phase_name}: avg {phase.average_ms:.1f}ms, "
            f"total {phase.total_ms:.1f}ms ({phase.count} measured)"
        )

    if result.detection_metrics is not None:
        dm = result.detection_metrics
        logger.info(f"  mAP@0.5: {dm.map_50:.4f}, mAP@0.5:0.95: {dm.map_50_95:.4f}")


def _log_detection_summary(summary: DetectionSummary) -> None:
    """検出サマリーをログ出力する.

    Args:
        summary: 検出サマリー.
    """
    logger.info("=== Detection Summary ===")
    logger.info(f"  Total images  : {summary.total_images}")
    logger.info(f"  Total detected: {summary.total_detections}")
    for cc in summary.per_class:
        logger.info(
            f"  {cc.name} : {cc.count} detections "
            f"({cc.images_with_detections} images, avg score: {cc.avg_score:.2f})"
        )
    if summary.images_without_detections > 0:
        logger.info(f"  No detections : {summary.images_without_detections} images")
