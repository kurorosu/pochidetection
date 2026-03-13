"""SSDLite MobileNetV3 推論スクリプト.

学習済み SSDLite モデルでフォルダ内の画像を一括推論する.
"""

from pathlib import Path
from typing import Any, NamedTuple

import torch
from PIL import Image
from torchvision.transforms import v2

from pochidetection.core.detection import Detection
from pochidetection.inference import SSDLiteOnnxBackend, SSDLitePyTorchBackend

try:
    from pochidetection.inference import SSDLiteTensorRTBackend

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False
from pochidetection.interfaces import IInferenceBackend
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
    config_path: str | None = None,
) -> None:
    """フォルダ内の画像を一括推論.

    Args:
        config: 設定辞書.
        image_dir: 推論対象の画像フォルダパス.
        model_dir: モデルディレクトリ, ONNX ファイル, または TensorRT エンジンのパス.
            None の場合は最新ワークスペースの best を使用.
        config_path: 設定ファイルのパス. 指定時は推論結果ディレクトリにコピーする.
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
    write_reports(config, image_files, all_predictions, ctx, model_path, config_path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_onnx_model(path: Path) -> bool:
    """パスが ONNX モデルファイルかを判定する.

    Args:
        path: 判定対象のパス.

    Returns:
        ONNX モデルファイルの場合 True.
    """
    return path.is_file() and path.suffix.lower() == ".onnx"


def _is_tensorrt_model(model_path: Path) -> bool:
    """モデルパスが TensorRT エンジンかどうかを判定する.

    Args:
        model_path: モデルのパス.

    Returns:
        .engine ファイルの場合 True.
    """
    return model_path.suffix.lower() == ".engine"


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
    num_classes = config["num_classes"]
    use_fp16 = config.get("use_fp16", False)
    image_size_cfg = config.get("image_size", {"height": 320, "width": 320})
    image_size = (image_size_cfg["height"], image_size_cfg["width"])
    nms_iou_threshold = config.get("nms_iou_threshold", 0.55)

    if _is_tensorrt_model(model_path):
        if not _TRT_AVAILABLE:
            raise ImportError(
                "tensorrt パッケージがインストールされていません. "
                "TensorRT バックエンドを使用するには TensorRT をインストールしてください."
            )
        logger.info("TensorRT backend selected")
        backend: IInferenceBackend = SSDLiteTensorRTBackend(
            engine_path=model_path,
            num_classes=num_classes,
            image_size=image_size,
            nms_iou_threshold=nms_iou_threshold,
        )
        return backend, "fp32", False

    if _is_onnx_model(model_path):
        logger.info("ONNX backend selected")
        backend = SSDLiteOnnxBackend(
            model_path=model_path,
            num_classes=num_classes,
            image_size=image_size,
            nms_iou_threshold=nms_iou_threshold,
            device=device,
        )
        return backend, "fp32", False

    model = SSDLiteModel(num_classes=num_classes, nms_iou_threshold=nms_iou_threshold)
    model.load(model_path)
    model.to(device)
    model.eval()

    fp16 = is_fp16_available(use_fp16, device)
    if fp16:
        model.half()
        logger.info("FP16 enabled")

    precision = "fp16" if fp16 else "fp32"
    return SSDLitePyTorchBackend(model), precision, use_fp16


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
        model_path: モデルのパス (ディレクトリ, ONNX ファイル, または TensorRT エンジン).

    Returns:
        構築済みのパイプラインコンテキスト.
    """
    device = config["device"]
    threshold = config["infer_score_threshold"]
    image_size_cfg = config.get("image_size", {"height": 320, "width": 320})
    image_size = (image_size_cfg["height"], image_size_cfg["width"])
    use_fp16 = config.get("use_fp16", False)

    if config.get("cudnn_benchmark", False) and device == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cudnn.benchmark enabled")

    backend, precision, use_fp16 = _create_backend(model_path, config)

    if _is_tensorrt_model(model_path):
        actual_device = "cuda"
        runtime_device = "cuda"
    else:
        actual_device = device
        runtime_device = device

    transform = v2.Compose(
        [
            v2.Resize(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    phased_timer = PhasedTimer(phases=SSDLitePipeline.PHASES, device=runtime_device)
    pipeline = SSDLitePipeline(
        backend=backend,
        transform=transform,
        image_size=image_size,
        device=runtime_device,
        threshold=threshold,
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
