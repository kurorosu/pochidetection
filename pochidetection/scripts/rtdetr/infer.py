"""RT-DETR 推論スクリプト.

学習済みRT-DETRモデルでフォルダ内の画像を一括推論する.
"""

from pathlib import Path
from typing import Any, NamedTuple

import torch
from PIL import Image
from transformers import RTDetrImageProcessor

from pochidetection.inference import OnnxBackend, PyTorchBackend

try:
    from pochidetection.inference import TensorRTBackend

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False
from pochidetection.core.detection import Detection
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.logging import LoggerManager
from pochidetection.models import RTDetrModel
from pochidetection.scripts.rtdetr.inference import (
    DetectionPipeline,
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
from pochidetection.utils.device import is_fp16_available
from pochidetection.utils.map_evaluator import MapEvaluator
from pochidetection.visualization import (
    ConfusionMatrixPlotter,
    LabelMapper,
    build_confusion_matrix,
)

logger = LoggerManager().get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def infer(
    config: dict[str, Any],
    image_dir: str,
    model_dir: str | None = None,
) -> None:
    """フォルダ内の画像を一括推論.

    Args:
        config: 設定辞書.
        image_dir: 推論対象の画像フォルダパス.
        model_dir: モデルディレクトリ. Noneの場合は最新ワークスペースのbestを使用.
    """
    model_path = _resolve_model_path(config, model_dir)
    if model_path is None:
        return

    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return

    image_files = [
        f for f in image_dir_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        logger.warning(f"No image files found in {image_dir}")
        return

    logger.info(f"Found {len(image_files)} images in {image_dir}")
    logger.info(f"Loading model from {model_path}")

    ctx = _setup_pipeline(config, model_path)
    logger.info(f"Results will be saved to {ctx.saver.output_dir}")

    all_predictions = _run_inference(image_files, ctx)
    _write_reports(config, image_files, all_predictions, ctx, model_path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class _PipelineContext(NamedTuple):
    """_setup_pipeline の戻り値."""

    pipeline: DetectionPipeline
    phased_timer: PhasedTimer
    visualizer: Visualizer
    saver: InferenceSaver
    label_mapper: LabelMapper | None
    class_names: list[str] | None
    actual_device: str
    precision: str


def _setup_pipeline(
    config: dict[str, Any],
    model_path: Path,
) -> _PipelineContext:
    """推論パイプラインの構築.

    Args:
        config: 設定辞書.
        model_path: モデルのパス.

    Returns:
        構築済みのパイプラインコンテキスト.
    """
    device = config["device"]
    threshold = config["infer_score_threshold"]
    nms_iou_threshold = config["nms_iou_threshold"]

    if config.get("cudnn_benchmark", False) and device == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cudnn.benchmark enabled")

    processor = _load_processor(model_path, config)
    backend, precision, use_fp16 = _create_backend(model_path, config)

    if _is_tensorrt_model(model_path):
        actual_device = "cuda"
        runtime_device = "cuda"
    elif _is_onnx_model(model_path):
        if not isinstance(backend, OnnxBackend):
            raise TypeError(f"Expected OnnxBackend, got {type(backend).__name__}")
        actual_device = (
            "cuda" if "CUDAExecutionProvider" in backend.active_providers else "cpu"
        )
        runtime_device = "cpu"
    else:
        actual_device = device
        runtime_device = device

    phased_timer = PhasedTimer(
        phases=DetectionPipeline.PHASES,
        device=runtime_device,
    )
    pipeline = DetectionPipeline(
        backend=backend,
        processor=processor,
        device=runtime_device,
        threshold=threshold,
        nms_iou_threshold=nms_iou_threshold,
        use_fp16=use_fp16,
        phased_timer=phased_timer,
    )

    class_names = config.get("class_names")
    label_mapper = LabelMapper(class_names) if class_names else None
    visualizer = Visualizer(label_mapper=label_mapper)

    is_single_file = _is_onnx_model(model_path) or _is_tensorrt_model(model_path)
    saver_base = model_path.parent if is_single_file else model_path
    saver = InferenceSaver(saver_base)

    return _PipelineContext(
        pipeline=pipeline,
        phased_timer=phased_timer,
        visualizer=visualizer,
        saver=saver,
        label_mapper=label_mapper,
        class_names=class_names,
        actual_device=actual_device,
        precision=precision,
    )


def _run_inference(
    image_files: list[Path],
    ctx: _PipelineContext,
) -> dict[str, list[Detection]]:
    """画像ループで推論を実行.

    Args:
        image_files: 推論対象の画像ファイルリスト.
        ctx: パイプラインコンテキスト.

    Returns:
        ファイル名をキー, 検出結果リストを値とする辞書.
    """
    all_predictions: dict[str, list[Detection]] = {}

    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")
        detections = ctx.pipeline.run(image)
        all_predictions[image_file.name] = detections
        result_image = ctx.visualizer.draw(image, detections)
        output_path = ctx.saver.save(result_image, image_file.name)

        inf_timer = ctx.phased_timer.get_timer("inference")
        logger.info(
            f"  {image_file.name} ({inf_timer.last_time_ms:.1f}ms) - "
            f"{len(detections)} objects -> {output_path.name}"
        )

    return all_predictions


def _write_reports(
    config: dict[str, Any],
    image_files: list[Path],
    all_predictions: dict[str, list[Detection]],
    ctx: _PipelineContext,
    model_path: Path,
) -> None:
    """レポート出力 (mAP, summary, CSV, confusion matrix, benchmark).

    Args:
        config: 設定辞書.
        image_files: 推論対象の画像ファイルリスト.
        all_predictions: ファイル名をキー, 検出結果リストを値とする辞書.
        ctx: パイプラインコンテキスト.
        model_path: モデルのパス.
    """
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


def _is_onnx_model(model_path: Path) -> bool:
    """モデルパスが ONNX ファイルかどうかを判定する.

    Args:
        model_path: モデルのパス.

    Returns:
        .onnx ファイルの場合 True.
    """
    return model_path.suffix.lower() == ".onnx"


def _is_tensorrt_model(model_path: Path) -> bool:
    """モデルパスが TensorRT エンジンかどうかを判定する.

    Args:
        model_path: モデルのパス.

    Returns:
        .engine ファイルの場合 True.
    """
    return model_path.suffix.lower() == ".engine"


def _load_processor(model_path: Path, config: dict[str, Any]) -> RTDetrImageProcessor:
    """画像前処理プロセッサを読み込む.

    ONNX / TensorRT モデルの場合, processor ファイルはモデルファイルと
    同じディレクトリから読み込みを試み, 見つからなければ config の
    model_name からフォールバックする.

    Args:
        model_path: モデルのパス.
        config: 設定辞書.

    Returns:
        RTDetrImageProcessor インスタンス.

    Raises:
        RuntimeError: processor が解決できない場合.
    """
    if not _is_onnx_model(model_path) and not _is_tensorrt_model(model_path):
        return RTDetrImageProcessor.from_pretrained(model_path)

    processor_dir = model_path.parent
    processor_config = processor_dir / "preprocessor_config.json"
    if processor_config.exists():
        logger.info(f"Loading processor from {processor_dir}")
        return RTDetrImageProcessor.from_pretrained(processor_dir)

    model_name = config.get("model_name")
    if model_name:
        logger.info(f"Loading processor from model_name: {model_name}")
        return RTDetrImageProcessor.from_pretrained(model_name)

    raise RuntimeError(
        f"RTDetrImageProcessor を解決できません. "
        f"{processor_dir} に preprocessor_config.json が見つからず, "
        f"config に model_name も指定されていません."
    )


def _create_backend(
    model_path: Path, config: dict[str, Any]
) -> tuple[IInferenceBackend, str, bool]:
    """モデルパスからバックエンドを生成する.

    Args:
        model_path: モデルのパス.
        config: 設定辞書.

    Returns:
        (backend, precision, use_fp16) のタプル.
    """
    device = config["device"]
    use_fp16 = config.get("use_fp16", False)

    if _is_tensorrt_model(model_path):
        if not _TRT_AVAILABLE:
            raise ImportError(
                "tensorrt パッケージがインストールされていません. "
                "TensorRT バックエンドを使用するには TensorRT をインストールしてください."
            )
        logger.info("TensorRT backend selected")
        return TensorRTBackend(model_path), "fp32", False

    if _is_onnx_model(model_path):
        logger.info("ONNX backend selected")
        return OnnxBackend(model_path, device=device), "fp32", False

    model = RTDetrModel(str(model_path))
    model.to(device)
    model.eval()

    fp16 = is_fp16_available(use_fp16, device)
    if fp16:
        model.half()
        logger.info("FP16 enabled")

    precision = "fp16" if fp16 else "fp32"
    return PyTorchBackend(model), precision, use_fp16


def _evaluate_map(
    config: dict[str, Any],
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


def _resolve_model_path(
    config: dict[str, Any],
    model_dir: str | None,
) -> Path | None:
    """モデルパスを解決.

    Args:
        config: 設定辞書.
        model_dir: 指定されたモデルディレクトリ.

    Returns:
        モデルパス. エラー時はNone.
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
        logger.error("No trained models found. Please run training first.")
        return None

    latest_workspace = Path(str(workspaces[-1]["path"]))
    model_path = latest_workspace / "best"

    if not model_path.exists():
        logger.error(
            f"Best model not found at {model_path}. Please run training first."
        )
        return None

    return model_path
