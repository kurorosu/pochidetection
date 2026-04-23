"""CLI batch 推論フローのオーケストレーション.

画像ディレクトリを入力とした一括推論 (``run_batch_inference``) と
関連するヘルパー (画像ファイル収集 / 推論ループ) を提供する.
"""

from pathlib import Path

from PIL import Image

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.core.detection import Detection
from pochidetection.logging import LoggerManager
from pochidetection.orchestration.reports import write_reports
from pochidetection.pipelines.context import InferenceContext
from pochidetection.pipelines.spec import resolve_and_build_pipeline
from pochidetection.utils.infer_debug import InferDebugConfig, save_infer_debug_image

logger = LoggerManager().get_logger(__name__)

__all__ = ["run_batch_inference"]


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def run_batch_inference(
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
    resolved = resolve_and_build_pipeline(config, model_dir, config_path)
    if resolved is None:
        return

    image_files = _collect_image_files(image_dir)
    if image_files is None:
        return

    ctx, config, config_path, model_path = resolved
    logger.info(f"Results will be saved to {ctx.saver.output_dir}")

    all_predictions = _run_inference(image_files, ctx, config, save_crop=save_crop)
    write_reports(config, image_files, all_predictions, ctx, model_path, config_path)


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


def _run_inference(
    image_files: list[Path],
    context: InferenceContext,
    config: DetectionConfigDict,
    *,
    save_crop: bool = True,
) -> dict[str, list[Detection]]:
    """画像ループで推論を実行.

    Args:
        image_files: 推論対象の画像ファイルリスト.
        context: パイプラインコンテキスト.
        config: 設定辞書. ``infer_debug_save_count`` / ``image_size`` /
            ``letterbox`` を debug 保存で参照する.
        save_crop: True の場合, 検出ボックスのクロップ画像を保存する.

    Returns:
        ファイル名をキー, 検出結果リストを値とする辞書.
    """
    all_predictions: dict[str, list[Detection]] = {}

    infer_debug = InferDebugConfig.from_config(config, context.saver.output_dir)

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

        detections = context.pipeline.run(image)
        all_predictions[image_file.name] = detections

        if save_crop:
            context.saver.save_crops(
                image, detections, image_file.name, context.label_mapper
            )

        result_image = context.visualizer.draw(image, detections, inplace=True)
        output_path = context.saver.save(result_image, image_file.name)

        inf_timer = context.phased_timer.get_timer("inference")
        logger.info(
            f"  {image_file.name} ({inf_timer.last_time_ms:.1f}ms) - "
            f"{len(detections)} objects -> {output_path.name}"
        )

    return all_predictions
