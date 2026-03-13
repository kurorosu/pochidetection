"""RT-DETR 推論スクリプト.

学習済みRT-DETRモデルでフォルダ内の画像を一括推論する.
"""

from pathlib import Path
from typing import Any, NamedTuple

import torch
from torchvision.transforms import v2
from transformers import RTDetrImageProcessor

from pochidetection.inference import RTDetrOnnxBackend, RTDetrPyTorchBackend

try:
    from pochidetection.inference import RTDetrTensorRTBackend

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.logging import LoggerManager
from pochidetection.models import RTDetrModel
from pochidetection.scripts.common import (
    InferenceSaver,
    Visualizer,
)
from pochidetection.scripts.common.inference import (
    create_backend,
)
from pochidetection.scripts.common.inference import infer as common_infer
from pochidetection.scripts.common.inference import (
    is_onnx_model,
    is_tensorrt_model,
)
from pochidetection.scripts.rtdetr.inference import (
    RTDetrPipeline,
)
from pochidetection.utils import PhasedTimer
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
        model_dir: モデルディレクトリ. Noneの場合は最新ワークスペースのbestを使用.
        config_path: 設定ファイルのパス. 指定時は推論結果ディレクトリにコピーする.
    """
    common_infer(config, image_dir, _setup_pipeline, model_dir, config_path)


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
    backend, precision, use_fp16 = create_backend(
        model_path,
        config,
        create_trt=lambda p: RTDetrTensorRTBackend(p),
        create_onnx=lambda p, d: RTDetrOnnxBackend(p, device=d),
        create_pytorch=_create_pytorch_backend,
        trt_available=_TRT_AVAILABLE,
    )

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

    if is_tensorrt_model(model_path):
        actual_device = "cuda"
        runtime_device = "cuda"
    elif is_onnx_model(model_path):
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

    is_single_file = is_onnx_model(model_path) or is_tensorrt_model(model_path)
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
    if not is_onnx_model(model_path) and not is_tensorrt_model(model_path):
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


def _create_pytorch_backend(
    model_path: Path, device: str, use_fp16: bool
) -> IInferenceBackend:
    """RT-DETR 用 PyTorch バックエンドを生成する.

    Args:
        model_path: モデルのパス.
        device: 推論デバイス.
        use_fp16: FP16 推論を使用するか.

    Returns:
        RTDetrPyTorchBackend インスタンス.
    """
    model = RTDetrModel(str(model_path))
    model.to(device)
    model.eval()

    if use_fp16:
        model.half()

    return RTDetrPyTorchBackend(model)
