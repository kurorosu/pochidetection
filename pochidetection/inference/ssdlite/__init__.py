"""SSDLite 推論バックエンドモジュール."""

from pochidetection.inference.ssdlite.onnx_backend import SSDLiteOnnxBackend
from pochidetection.inference.ssdlite.pytorch_backend import SSDLitePyTorchBackend

__all__ = [
    "SSDLiteOnnxBackend",
    "SSDLitePyTorchBackend",
]

try:
    from pochidetection.inference.ssdlite.tensorrt_backend import (
        SSDLiteTensorRTBackend,
    )

    __all__.append("SSDLiteTensorRTBackend")
except ImportError:
    pass
