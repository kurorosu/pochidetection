"""推論の共通ロジック (CLI batch 用).

``infer`` (batch 画像ディレクトリ推論) と関連するレポート出力 / サマリーログを
提供する. pipeline 構築部分は ``pipelines/spec.py`` 等に分離済み.
"""

from pathlib import Path

from PIL import Image

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.core.detection import Detection
from pochidetection.logging import LoggerManager
from pochidetection.pipelines.context import _InferenceContext
from pochidetection.pipelines.spec import resolve_and_setup_pipeline
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
from pochidetection.utils.infer_debug import InferDebugConfig, save_infer_debug_image
from pochidetection.utils.map_evaluator import MapEvaluator
from pochidetection.visualization import (
    ConfusionMatrixPlotter,
    build_confusion_matrix,
)

logger = LoggerManager().get_logger(__name__)

__all__ = ["infer"]


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


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
