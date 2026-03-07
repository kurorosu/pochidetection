"""SSDLite MobileNetV3 推論スクリプト.

学習済み SSDLite モデルでフォルダ内の画像を一括推論する.
"""

from pathlib import Path
from typing import Any, NamedTuple

import torch
from PIL import Image
from torchvision import transforms

from pochidetection.core.detection import Detection
from pochidetection.logging import LoggerManager
from pochidetection.models import SSDLiteModel
from pochidetection.scripts.rtdetr.inference import (
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
PHASES = ["preprocess", "inference", "postprocess"]


def infer(
    config: dict[str, Any],
    image_dir: str,
    model_dir: str | None = None,
) -> None:
    """フォルダ内の画像を一括推論.

    Args:
        config: 設定辞書.
        image_dir: 推論対象の画像フォルダパス.
        model_dir: モデルディレクトリ. None の場合は最新ワークスペースの best を使用.
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

    model: SSDLiteModel
    transform: transforms.Compose
    image_size: tuple[int, int]
    threshold: float
    device: str
    precision: str
    phased_timer: PhasedTimer
    visualizer: Visualizer
    saver: InferenceSaver
    label_mapper: LabelMapper | None
    class_names: list[str] | None


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
    num_classes = config["num_classes"]
    image_size_cfg = config.get("image_size", {"height": 320, "width": 320})
    image_size = (image_size_cfg["height"], image_size_cfg["width"])
    use_fp16 = config.get("use_fp16", False)

    if config.get("cudnn_benchmark", False) and device == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cudnn.benchmark enabled")

    model = SSDLiteModel(
        num_classes=num_classes,
        model_path=model_path,
    )
    model.to(device)
    model.eval()

    fp16 = is_fp16_available(use_fp16, device)
    if fp16:
        model.half()
        logger.info("FP16 enabled")

    precision = "fp16" if fp16 else "fp32"

    transform = transforms.Compose([transforms.ToTensor()])

    phased_timer = PhasedTimer(phases=PHASES, device=device)

    class_names = config.get("class_names")
    label_mapper = LabelMapper(class_names) if class_names else None
    visualizer = Visualizer(label_mapper=label_mapper)
    saver = InferenceSaver(model_path)

    return _PipelineContext(
        model=model,
        transform=transform,
        image_size=image_size,
        threshold=threshold,
        device=device,
        precision=precision,
        phased_timer=phased_timer,
        visualizer=visualizer,
        saver=saver,
        label_mapper=label_mapper,
        class_names=class_names,
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
    target_h, target_w = ctx.image_size

    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")
        orig_w, orig_h = image.size

        # 前処理
        with ctx.phased_timer.measure("preprocess"):
            image_resized = image.resize((target_w, target_h))
            pixel_values = ctx.transform(image_resized).unsqueeze(0).to(ctx.device)

        # 推論
        with ctx.phased_timer.measure("inference"):
            with torch.no_grad():
                outputs = ctx.model(pixel_values)

        # 後処理
        with ctx.phased_timer.measure("postprocess"):
            detections = _postprocess(
                outputs["predictions"][0],
                ctx.threshold,
                orig_w,
                orig_h,
                target_w,
                target_h,
            )

        all_predictions[image_file.name] = detections
        result_image = ctx.visualizer.draw(image, detections)
        output_path = ctx.saver.save(result_image, image_file.name)

        inf_timer = ctx.phased_timer.get_timer("inference")
        logger.info(
            f"  {image_file.name} ({inf_timer.last_time_ms:.1f}ms) - "
            f"{len(detections)} objects -> {output_path.name}"
        )

    return all_predictions


def _postprocess(
    pred: dict[str, torch.Tensor],
    threshold: float,
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
) -> list[Detection]:
    """予測結果をフィルタリングし, 元画像座標に変換.

    Args:
        pred: モデル出力 (boxes, scores, labels).
        threshold: 検出信頼度閾値.
        orig_w: 元画像の幅.
        orig_h: 元画像の高さ.
        target_w: リサイズ後の幅.
        target_h: リサイズ後の高さ.

    Returns:
        検出結果のリスト.
    """
    mask = pred["scores"] >= threshold
    boxes = pred["boxes"][mask]
    scores = pred["scores"][mask]
    labels = pred["labels"][mask]

    # リサイズ座標 → 元画像座標
    scale_x = orig_w / target_w
    scale_y = orig_h / target_h
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y

    return [
        Detection(
            box=box.tolist(),
            score=score.item(),
            label=label.item(),
        )
        for box, score, label in zip(boxes, scores, labels)
    ]


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
        device=ctx.device,
        precision=ctx.precision,
        model_path=str(model_path),
        detection_metrics=detection_metrics,
    )

    json_path = write_benchmark_result(ctx.saver.output_dir, result)
    logger.info(f"Benchmark result saved to {json_path}")

    _log_benchmark_summary(result)
    logger.info(f"Results saved to {ctx.saver.output_dir}")


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
