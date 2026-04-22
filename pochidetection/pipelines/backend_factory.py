"""推論 backend (PyTorch / ONNX / TensorRT) の生成ロジック共通化.

アーキ別 (RT-DETR / SSDLite / SSD300) に共通する backend 生成の分岐
(モデルパス → 種別判定 → 対応 factory 呼出 → precision / fp16 決定) を
``create_backend`` に集約する. 具象 backend の生成はコールバックに委譲し,
アーキ側は ``ArchitectureSpec.backends`` に 3 種 factory を渡すだけで良い.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.logging import LoggerManager
from pochidetection.pipelines.model_path import is_onnx_model, is_tensorrt_model
from pochidetection.utils.device import is_fp16_available

__all__ = ["create_backend"]

logger = LoggerManager().get_logger(__name__)


# バックエンドファクトリコールバックの型
_CreateTrtFn = Callable[[Path], IInferenceBackend[Any]]
_CreateOnnxFn = Callable[[Path, str], IInferenceBackend[Any]]
_CreatePytorchFn = Callable[[Path, str, bool], IInferenceBackend[Any]]


def create_backend(
    model_path: Path,
    config: DetectionConfigDict,
    create_trt: _CreateTrtFn,
    create_onnx: _CreateOnnxFn,
    create_pytorch: _CreatePytorchFn,
    trt_available: bool = False,
) -> tuple[IInferenceBackend[Any], str, bool]:
    """モデルパスからバックエンドを生成する.

    TensorRT / ONNX / PyTorch の分岐ロジックを共通化し,
    具象バックエンドの生成はコールバックに委譲する.

    Args:
        model_path: モデルのパス.
        config: 設定辞書.
        create_trt: TensorRT バックエンド生成コールバック.
            (model_path,) を受け取り IInferenceBackend を返す.
        create_onnx: ONNX バックエンド生成コールバック.
            (model_path, device) を受け取り IInferenceBackend を返す.
        create_pytorch: PyTorch バックエンド生成コールバック.
            (model_path, device, use_fp16) を受け取り IInferenceBackend を返す.
            FP16 適用 (model.half()) はコールバック側で行う.
        trt_available: TensorRT が利用可能かどうか.

    Returns:
        (backend, precision, use_fp16) のタプル.
    """
    device = config["device"]
    use_fp16 = config.get("use_fp16", False)

    if is_tensorrt_model(model_path):
        if not trt_available:
            raise ImportError(
                "tensorrt パッケージがインストールされていません. "
                "TensorRT バックエンドを使用するには TensorRT をインストールしてください."
            )
        logger.info("TensorRT backend selected")
        return create_trt(model_path), "fp32", False

    if is_onnx_model(model_path):
        logger.info("ONNX backend selected")
        return create_onnx(model_path, device), "fp32", False

    fp16 = is_fp16_available(use_fp16, device)
    backend = create_pytorch(model_path, device, fp16)

    if fp16:
        logger.info("FP16 enabled")

    precision = "fp16" if fp16 else "fp32"
    return backend, precision, use_fp16
