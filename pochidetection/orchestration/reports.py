"""推論結果のレポート出力.

``write_reports`` を入口として mAP 評価 / 検出サマリー / CSV / 混同行列 /
ベンチマーク結果を出力し, ログサマリーを表示する.
"""

from pathlib import Path

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.core.detection import Detection
from pochidetection.logging import LoggerManager
from pochidetection.pipelines.context import InferenceContext
from pochidetection.reporting import (
    DetectionSummary,
    build_detection_results,
    build_detection_summary,
    write_detection_results_csv,
    write_detection_summary,
)
from pochidetection.utils import (
    BenchmarkResult,
    DetectionMetrics,
    build_benchmark_result,
    write_benchmark_result,
)
from pochidetection.utils.config_loader import ConfigLoader
from pochidetection.utils.map_evaluator import MapEvaluator
from pochidetection.visualization import (
    ConfusionMatrixPlotter,
    build_confusion_matrix,
)

logger = LoggerManager().get_logger(__name__)

__all__ = ["write_reports"]


def write_reports(
    config: DetectionConfigDict,
    image_files: list[Path],
    all_predictions: dict[str, list[Detection]],
    context: InferenceContext,
    model_path: Path,
    config_path: str | None = None,
) -> None:
    """レポート出力 (mAP, summary, CSV, confusion matrix, benchmark).

    Args:
        config: 設定辞書.
        image_files: 推論対象の画像ファイルリスト.
        all_predictions: ファイル名をキー, 検出結果リストを値とする辞書.
        context: 推論コンテキスト.
        model_path: モデルのパス.
        config_path: 設定ファイルのパス. 指定時は推論結果ディレクトリにコピーする.
    """
    if config_path is not None:
        _save_config(config, config_path, context.saver.output_dir)

    detection_metrics = _evaluate_map(config, all_predictions)

    summary = build_detection_summary(all_predictions, context.label_mapper)
    summary_path = write_detection_summary(context.saver.output_dir, summary)
    logger.info(f"Detection summary saved to {summary_path}")
    _log_detection_summary(summary)

    annotation_path_str = config.get("annotation_path")
    annotation_path = Path(annotation_path_str) if annotation_path_str else None
    if annotation_path is not None and not annotation_path.exists():
        logger.warning(f"Annotation file not found: {annotation_path}")
        annotation_path = None

    csv_rows = build_detection_results(
        predictions=all_predictions,
        label_mapper=context.label_mapper,
        annotation_path=annotation_path,
    )
    csv_path = write_detection_results_csv(context.saver.output_dir, csv_rows)
    logger.info(f"Detection results CSV saved to {csv_path}")

    if annotation_path is not None and context.class_names is not None:
        cm = build_confusion_matrix(
            predictions=all_predictions,
            annotation_path=annotation_path,
            class_names=context.class_names,
        )
        cm_plotter = ConfusionMatrixPlotter(cm, context.class_names)
        cm_path = context.saver.output_dir / "confusion_matrix.html"
        cm_plotter.plot(cm_path)
        logger.info(f"Confusion matrix saved to {cm_path}")

    result = build_benchmark_result(
        phased_timer=context.phased_timer,
        num_images=len(image_files),
        device=context.actual_device,
        precision=context.precision,
        model_path=str(model_path),
        detection_metrics=detection_metrics,
    )

    json_path = write_benchmark_result(context.saver.output_dir, result)
    logger.info(f"Benchmark result saved to {json_path}")

    _log_benchmark_summary(result)
    logger.info(f"Results saved to {context.saver.output_dir}")


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
