"""RT-DETR 推論スクリプト.

学習済みRT-DETRモデルでフォルダ内の画像を一括推論する.
"""

from pathlib import Path
from typing import Any

from PIL import Image

from pochidetection.logging import LoggerManager
from pochidetection.scripts.rtdetr.inference import (
    Detector,
    InferenceSaver,
    Visualizer,
)
from pochidetection.utils import InferenceTimer, WorkspaceManager
from pochidetection.visualization import LabelMapper

logger = LoggerManager().get_logger(__name__)

# サポートする画像拡張子
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def infer(
    config: dict[str, Any],
    image_dir: str,
    threshold: float = 0.5,
    model_dir: str | None = None,
) -> None:
    """フォルダ内の画像を一括推論.

    Args:
        config: 設定辞書.
        image_dir: 推論対象の画像フォルダパス.
        threshold: 検出信頼度閾値.
        model_dir: モデルディレクトリ. Noneの場合は最新ワークスペースのbestを使用.
    """
    device = config["device"]

    # モデルディレクトリの決定
    model_path = _resolve_model_path(config, model_dir)
    if model_path is None:
        return

    # 画像ファイルを取得
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

    # コンポーネント初期化
    timer = InferenceTimer(device=device)
    detector = Detector(model_path, device=device, threshold=threshold, timer=timer)
    class_names = config.get("class_names")
    label_mapper = LabelMapper(class_names) if class_names else None
    visualizer = Visualizer(label_mapper=label_mapper)
    saver = InferenceSaver(model_path)

    logger.info(f"Results will be saved to {saver.output_dir}")

    # 一括推論
    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")

        # 検出
        detections = detector.detect(image)

        # 描画
        result_image = visualizer.draw(image, detections)

        # 保存
        output_path = saver.save(result_image, image_file.name)

        logger.info(
            f"  {image_file.name} ({timer.last_time_ms:.1f}ms) - "
            f"{len(detections)} objects -> {output_path.name}"
        )

    # サマリー出力
    total_sec = timer.total_time_ms / 1000
    logger.info(
        f"Inference completed: {timer.count} images "
        f"(1st skipped for warmup), "
        f"avg {timer.average_time_ms:.1f}ms/image, total {total_sec:.2f}s"
    )
    logger.info(f"Results saved to {saver.output_dir}")


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

    # 最新のワークスペースからベストモデルを探す
    workspace_manager = WorkspaceManager(config["work_dir"])
    workspaces = workspace_manager.get_available_workspaces()

    if not workspaces:
        logger.error("No trained models found. Please run training first.")
        return None

    # 最新のワークスペースのbestディレクトリを使用
    latest_workspace = Path(str(workspaces[-1]["path"]))
    model_path = latest_workspace / "best"

    if not model_path.exists():
        logger.error(
            f"Best model not found at {model_path}. Please run training first."
        )
        return None

    return model_path
