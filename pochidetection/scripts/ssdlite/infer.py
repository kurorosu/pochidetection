"""SSDLite MobileNetV3 推論スクリプト.

学習済み SSDLite モデルでフォルダ内の画像を一括推論する.
"""

from pathlib import Path

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.inference import (
    SSDLiteOnnxBackend,
    SsdPyTorchBackend,
    build_pytorch_backend,
)

try:
    from pochidetection.inference import SSDLiteTensorRTBackend

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False
from pochidetection.logging import LoggerManager
from pochidetection.models import SSDLiteModel
from pochidetection.orchestration import run_batch_inference
from pochidetection.pipelines import SsdPipeline
from pochidetection.pipelines.context import PipelineContext
from pochidetection.pipelines.spec import (
    ArchitectureSpec,
    BackendFactories,
    build_pipeline_from_spec,
)

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
        model_dir: モデルディレクトリ, ONNX ファイル, または TensorRT エンジンのパス.
            None の場合は最新ワークスペースの best を使用.
        config_path: 設定ファイルのパス. 指定時は推論結果ディレクトリにコピーする.
        save_crop: True の場合, 検出ボックスのクロップ画像を保存する.
    """
    run_batch_inference(
        config,
        image_dir,
        model_dir,
        config_path,
        save_crop=save_crop,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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

    model = SSDLiteModel(num_classes=num_classes, nms_iou_threshold=nms_iou_threshold)
    model.load(model_path)

    return build_pytorch_backend(model, SsdPyTorchBackend, device, use_fp16)


def build_pipeline(
    config: DetectionConfigDict,
    model_path: Path,
) -> PipelineContext:
    """推論パイプラインの構築 (``ArchitectureSpec`` 経由で共通化)."""
    num_classes = config["num_classes"]
    image_size_cfg = config.get("image_size", {"height": 320, "width": 320})
    image_size = (int(image_size_cfg["height"]), int(image_size_cfg["width"]))
    nms_iou_threshold = config.get("nms_iou_threshold", 0.5)

    spec = ArchitectureSpec(
        pipeline_cls=SsdPipeline,
        backends=BackendFactories(
            pytorch=lambda p, d, fp16: _create_pytorch_backend(p, d, fp16, config),
            onnx=lambda p, d: SSDLiteOnnxBackend(
                model_path=p,
                num_classes=num_classes,
                image_size=image_size,
                nms_iou_threshold=nms_iou_threshold,
                device=d,
            ),
            tensorrt=lambda p: SSDLiteTensorRTBackend(
                engine_path=p,
                num_classes=num_classes,
                image_size=image_size,
                nms_iou_threshold=nms_iou_threshold,
            ),
            trt_available=_TRT_AVAILABLE,
        ),
        build_pipeline_kwargs=lambda cfg, hw, _processor: {"image_size": hw},
        default_image_size=(320, 320),
    )
    context = build_pipeline_from_spec(spec, config, model_path)
    return context
