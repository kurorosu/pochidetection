"""SSD300 VGG16 推論スクリプト.

学習済み SSD300 モデルでフォルダ内の画像を一括推論する.
"""

from pathlib import Path

import torch
from torchvision.transforms import v2

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.inference import SsdPyTorchBackend
from pochidetection.logging import LoggerManager
from pochidetection.models import SSD300Model
from pochidetection.scripts.common.inference import (
    PipelineContext,
    build_pipeline_context,
    create_backend,
)
from pochidetection.scripts.common.inference import infer as common_infer
from pochidetection.scripts.common.inference import (
    resolve_device,
    setup_cudnn_benchmark,
)
from pochidetection.scripts.ssd.inference import SsdPipeline
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
        model_dir: モデルディレクトリのパス.
            None の場合は最新ワークスペースの best を使用.
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


def _unsupported_trt(model_path: Path) -> SsdPyTorchBackend:
    """SSD300 は TensorRT バックエンド未対応."""
    msg = "SSD300 TensorRT backend is not supported"
    raise NotImplementedError(msg)


def _unsupported_onnx(model_path: Path, device: str) -> SsdPyTorchBackend:
    """SSD300 は ONNX バックエンド未対応."""
    msg = "SSD300 ONNX backend is not supported"
    raise NotImplementedError(msg)


def _create_pytorch_backend(
    model_path: Path, device: str, use_fp16: bool, config: DetectionConfigDict
) -> SsdPyTorchBackend:
    """モデル固有の PyTorch バックエンドを生成する.

    Args:
        model_path: モデルのパス.
        device: 推論デバイス.
        use_fp16: FP16 推論を使用するか.
        config: 設定辞書.

    Returns:
        SsdPyTorchBackend インスタンス.
    """
    num_classes = config["num_classes"]
    nms_iou_threshold = config.get("nms_iou_threshold", 0.5)

    model = SSD300Model(num_classes=num_classes, nms_iou_threshold=nms_iou_threshold)
    model.load(model_path)
    model.to(device)
    model.eval()

    if use_fp16:
        model.half()

    return SsdPyTorchBackend(model)


def _setup_pipeline(
    config: DetectionConfigDict,
    model_path: Path,
) -> PipelineContext:
    """推論パイプラインの構築.

    Args:
        config: 設定辞書.
        model_path: モデルのパス (ディレクトリ).

    Returns:
        構築済みのパイプラインコンテキスト.
    """
    threshold = config["infer_score_threshold"]
    image_size_cfg = config.get("image_size", {"height": 300, "width": 300})
    image_size = (image_size_cfg["height"], image_size_cfg["width"])

    setup_cudnn_benchmark(config)

    backend, precision, use_fp16 = create_backend(
        model_path,
        config,
        create_trt=_unsupported_trt,
        create_onnx=_unsupported_onnx,
        create_pytorch=lambda p, d, fp16: _create_pytorch_backend(p, d, fp16, config),
        trt_available=False,
    )

    actual_device, runtime_device = resolve_device(model_path, config, backend)

    transform = v2.Compose(
        [
            v2.Resize(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    phased_timer = PhasedTimer(phases=SsdPipeline.PHASES, device=runtime_device)
    pipeline = SsdPipeline(
        backend=backend,
        transform=transform,
        image_size=image_size,
        device=runtime_device,
        threshold=threshold,
        use_fp16=use_fp16,
        phased_timer=phased_timer,
    )

    return build_pipeline_context(
        pipeline=pipeline,
        phased_timer=phased_timer,
        config=config,
        model_path=model_path,
        actual_device=actual_device,
        precision=precision,
    )
