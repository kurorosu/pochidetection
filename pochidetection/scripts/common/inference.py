"""推論の共通ロジック.

RT-DETR と SSDLite で共有される推論エントリ, レポート出力,
ベンチマークサマリーのロジックを提供する.
"""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from pochidetection.core.detection import Detection
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.logging import LoggerManager
from pochidetection.scripts.common import (
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
from pochidetection.utils.map_evaluator import MapEvaluator
from pochidetection.visualization import (
    ConfusionMatrixPlotter,
    LabelMapper,
    build_confusion_matrix,
)

logger = LoggerManager().get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


class InferenceContext(Protocol):
    """推論コンテキストのプロトコル."""

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


def resolve_model_path(
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


def collect_image_files(image_dir: str) -> list[Path] | None:
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
        f for f in image_dir_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        logger.warning(f"No image files found in {image_dir}")
        return None

    logger.info(f"Found {len(image_files)} images in {image_dir}")
    return image_files


# セットアップコールバックの型
SetupPipelineFn = Callable[[dict[str, Any], Path], InferenceContext]


def infer(
    config: dict[str, Any],
    image_dir: str,
    setup_pipeline: SetupPipelineFn,
    model_dir: str | None = None,
    config_path: str | None = None,
) -> None:
    """フォルダ内の画像を一括推論.

    Args:
        config: 設定辞書.
        image_dir: 推論対象の画像フォルダパス.
        setup_pipeline: パイプライン構築コールバック.
            (config, model_path) を受け取り InferenceContext を返す.
        model_dir: モデルディレクトリ. None の場合は最新ワークスペースの best を使用.
        config_path: 設定ファイルのパス. 指定時は推論結果ディレクトリにコピーする.
    """
    model_path = resolve_model_path(config, model_dir)
    if model_path is None:
        return

    image_files = collect_image_files(image_dir)
    if image_files is None:
        return

    logger.info(f"Loading model from {model_path}")

    ctx = setup_pipeline(config, model_path)
    logger.info(f"Results will be saved to {ctx.saver.output_dir}")

    all_predictions = run_inference(image_files, ctx)
    write_reports(config, image_files, all_predictions, ctx, model_path, config_path)


def run_inference(
    image_files: list[Path],
    ctx: InferenceContext,
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


def write_reports(
    config: dict[str, Any],
    image_files: list[Path],
    all_predictions: dict[str, list[Detection]],
    ctx: InferenceContext,
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
        _save_config(config_path, ctx.saver.output_dir)

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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _save_config(config_path: str, output_dir: Path) -> None:
    """設定ファイルを推論結果ディレクトリにコピーする.

    Args:
        config_path: 設定ファイルのパス.
        output_dir: 推論結果の出力ディレクトリ.
    """
    src = Path(config_path)
    if not src.exists():
        logger.warning(f"Config file not found: {config_path}")
        return
    dst = output_dir / src.name
    shutil.copy2(src, dst)
    logger.info(f"Config saved to {dst}")


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
