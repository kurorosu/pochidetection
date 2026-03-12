"""RT-DETR 推論スクリプト.

学習済みRT-DETRモデルでフォルダ内の画像を一括推論する.
"""

from pathlib import Path
from typing import Any, NamedTuple

import torch
from PIL import Image
from torchvision.transforms import v2
from transformers import RTDetrImageProcessor

from pochidetection.inference import RTDetrOnnxBackend, RTDetrPyTorchBackend

try:
    from pochidetection.inference import RTDetrTensorRTBackend

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False
from pochidetection.core.detection import Detection
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.logging import LoggerManager
from pochidetection.models import RTDetrModel
from pochidetection.scripts.common import (
    InferenceSaver,
    Visualizer,
)
from pochidetection.scripts.common.inference import (
    collect_image_files,
    resolve_model_path,
    write_reports,
)
from pochidetection.scripts.rtdetr.inference import (
    RTDetrPipeline,
)
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
        model_dir: モデルディレクトリ. Noneの場合は最新ワークスペースのbestを使用.
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

    pipeline: RTDetrPipeline
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

    image_size = (
        int(config["image_size"]["height"]),
        int(config["image_size"]["width"]),
    )
    transform = v2.Compose(
        [
            v2.Resize(image_size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    if _is_tensorrt_model(model_path):
        actual_device = "cuda"
        runtime_device = "cuda"
    elif _is_onnx_model(model_path):
        if not isinstance(backend, RTDetrOnnxBackend):
            raise TypeError(f"Expected RTDetrOnnxBackend, got {type(backend).__name__}")
        actual_device = (
            "cuda" if "CUDAExecutionProvider" in backend.active_providers else "cpu"
        )
        runtime_device = "cpu"
    else:
        actual_device = device
        runtime_device = device

    phased_timer = PhasedTimer(
        phases=RTDetrPipeline.PHASES,
        device=runtime_device,
    )
    pipeline = RTDetrPipeline(
        backend=backend,
        processor=processor,
        transform=transform,
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
        return RTDetrTensorRTBackend(model_path), "fp32", False

    if _is_onnx_model(model_path):
        logger.info("ONNX backend selected")
        return RTDetrOnnxBackend(model_path, device=device), "fp32", False

    model = RTDetrModel(str(model_path))
    model.to(device)
    model.eval()

    fp16 = is_fp16_available(use_fp16, device)
    if fp16:
        model.half()
        logger.info("FP16 enabled")

    precision = "fp16" if fp16 else "fp32"
    return RTDetrPyTorchBackend(model), precision, use_fp16
