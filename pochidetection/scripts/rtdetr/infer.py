"""RT-DETR 推論スクリプト.

学習済みRT-DETRモデルでフォルダ内の画像を一括推論する.
"""

from pathlib import Path

import torch
from torchvision.transforms import v2
from transformers import RTDetrImageProcessor

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.inference import RTDetrOnnxBackend, RTDetrPyTorchBackend

try:
    from pochidetection.inference import RTDetrTensorRTBackend

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False
from pochidetection.logging import LoggerManager
from pochidetection.models import RTDetrModel
from pochidetection.pipelines import RTDetrPipeline
from pochidetection.pipelines.builder import (
    PRETRAINED,
    PipelineContext,
    build_pipeline_context,
    create_backend,
)
from pochidetection.pipelines.builder import infer as common_infer
from pochidetection.pipelines.builder import (
    is_onnx_model,
    is_tensorrt_model,
    resolve_device,
    resolve_pipeline_mode,
    setup_cudnn_benchmark,
)
from pochidetection.utils import PhasedTimer

logger = LoggerManager().get_logger(__name__)


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
        model_dir: モデルディレクトリ. Noneの場合は最新ワークスペースのbestを使用.
        config_path: 設定ファイルのパス. 指定時は推論結果ディレクトリにコピーする.
        save_crop: True の場合, 検出ボックスのクロップ画像を保存する.
    """
    common_infer(
        config,
        image_dir,
        model_dir,
        config_path,
        save_crop=save_crop,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _setup_pipeline(
    config: DetectionConfigDict,
    model_path: Path,
) -> PipelineContext:
    """推論パイプラインの構築.

    Args:
        config: 設定辞書.
        model_path: モデルのパス.

    Returns:
        構築済みのパイプラインコンテキスト.
    """
    threshold = config["infer_score_threshold"]
    nms_iou_threshold = config["nms_iou_threshold"]

    setup_cudnn_benchmark(config)

    processor = _load_processor(model_path, config)
    backend, precision, use_fp16 = create_backend(
        model_path,
        config,
        create_trt=lambda p: RTDetrTensorRTBackend(p),
        create_onnx=lambda p, d: RTDetrOnnxBackend(p, device=d),
        create_pytorch=lambda p, d, fp16: _create_pytorch_backend(p, d, fp16, config),
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

    actual_device, runtime_device = resolve_device(model_path, config, backend)
    pipeline_mode = resolve_pipeline_mode(config.get("pipeline_mode"), model_path)

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
        pipeline_mode=pipeline_mode,
        image_size=image_size,
        letterbox=config.get("letterbox", True),
    )

    return build_pipeline_context(
        pipeline=pipeline,
        phased_timer=phased_timer,
        config=config,
        model_path=model_path,
        actual_device=actual_device,
        precision=precision,
    )


def _load_processor(
    model_path: Path, config: DetectionConfigDict
) -> RTDetrImageProcessor:
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
    local_files_only = config.get("local_files_only", False)

    if model_path == PRETRAINED:
        pretrained_name = config.get("model_name", "PekingU/rtdetr_r50vd")
        return RTDetrImageProcessor.from_pretrained(
            pretrained_name, local_files_only=local_files_only
        )

    if not is_onnx_model(model_path) and not is_tensorrt_model(model_path):
        return RTDetrImageProcessor.from_pretrained(model_path)

    processor_dir = model_path.parent
    processor_config = processor_dir / "preprocessor_config.json"
    if processor_config.exists():
        logger.info(f"Loading processor from {processor_dir}")
        return RTDetrImageProcessor.from_pretrained(processor_dir)

    fallback_name: str | None = config.get("model_name")
    if fallback_name:
        logger.info(f"Loading processor from model_name: {fallback_name}")
        return RTDetrImageProcessor.from_pretrained(fallback_name)

    raise RuntimeError(
        f"RTDetrImageProcessor を解決できません. "
        f"{processor_dir} に preprocessor_config.json が見つからず, "
        f"config に model_name も指定されていません."
    )


def _create_pytorch_backend(
    model_path: Path,
    device: str,
    use_fp16: bool,
    config: DetectionConfigDict,
) -> RTDetrPyTorchBackend:
    """RT-DETR 用 PyTorch バックエンドを生成する.

    Args:
        model_path: モデルのパス.
        device: 推論デバイス.
        use_fp16: FP16 推論を使用するか.
        config: 設定辞書.

    Returns:
        RTDetrPyTorchBackend インスタンス.
    """
    local_files_only = config.get("local_files_only", False)

    if model_path == PRETRAINED:
        model_name = config.get("model_name", "PekingU/rtdetr_r50vd")
        model = RTDetrModel(model_name=model_name, local_files_only=local_files_only)
    else:
        model = RTDetrModel(str(model_path))
    model.to(device)
    model.eval()

    if use_fp16:
        model.half()

    return RTDetrPyTorchBackend(model)
