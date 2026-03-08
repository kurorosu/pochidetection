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
from pochidetection.scripts.common import (
    InferenceSaver,
    Visualizer,
)
from pochidetection.scripts.common.inference import (
    collect_image_files,
    resolve_model_path,
    write_reports,
)
from pochidetection.scripts.ssdlite.inference import SSDLitePipeline
from pochidetection.utils import PhasedTimer
from pochidetection.utils.device import is_fp16_available
from pochidetection.visualization import LabelMapper

logger = LoggerManager().get_logger(__name__)


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
    model_path = resolve_model_path(config, model_dir)
    if model_path is None:
        return

    image_files = collect_image_files(image_dir)
    if image_files is None:
        return

    logger.info(f"Loading model from {model_path}")

    ctx = _setup_pipeline(config, model_path)
    logger.info(f"Results will be saved to {ctx.saver.output_dir}")

    all_predictions = _run_inference(image_files, ctx)
    write_reports(config, image_files, all_predictions, ctx, model_path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class _PipelineContext(NamedTuple):
    """_setup_pipeline の戻り値."""

    pipeline: SSDLitePipeline
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
    num_classes = config["num_classes"]
    image_size_cfg = config.get("image_size", {"height": 320, "width": 320})
    image_size = (image_size_cfg["height"], image_size_cfg["width"])
    use_fp16 = config.get("use_fp16", False)

    if config.get("cudnn_benchmark", False) and device == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cudnn.benchmark enabled")

    model = SSDLiteModel(num_classes=num_classes)
    model.load(model_path)
    model.to(device)
    model.eval()

    fp16 = is_fp16_available(use_fp16, device)
    if fp16:
        model.half()
        logger.info("FP16 enabled")

    precision = "fp16" if fp16 else "fp32"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    phased_timer = PhasedTimer(phases=SSDLitePipeline.PHASES, device=device)
    pipeline = SSDLitePipeline(
        model=model,
        transform=transform,
        image_size=image_size,
        device=device,
        threshold=threshold,
        use_fp16=use_fp16,
        phased_timer=phased_timer,
    )

    class_names = config.get("class_names")
    label_mapper = LabelMapper(class_names) if class_names else None
    visualizer = Visualizer(label_mapper=label_mapper)
    saver = InferenceSaver(model_path)

    return _PipelineContext(
        pipeline=pipeline,
        phased_timer=phased_timer,
        visualizer=visualizer,
        saver=saver,
        label_mapper=label_mapper,
        class_names=class_names,
        actual_device=device,
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
