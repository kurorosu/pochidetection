"""SSDLite 推論バックエンドモジュール."""

from pochidetection.inference.ssdlite.onnx_backend import SSDLiteOnnxBackend
from pochidetection.inference.ssdlite.pytorch_backend import SSDLitePyTorchBackend

__all__ = [
    "SSDLiteOnnxBackend",
    "SSDLitePyTorchBackend",
]
