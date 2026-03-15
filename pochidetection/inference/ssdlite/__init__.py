"""SSDLite 推論バックエンドモジュール."""

from pochidetection.inference.ssd import SsdPyTorchBackend
from pochidetection.inference.ssdlite.onnx_backend import SSDLiteOnnxBackend

__all__ = [
    "SSDLiteOnnxBackend",
    "SsdPyTorchBackend",
]

try:
    from pochidetection.inference.ssdlite.tensorrt_backend import (
        SSDLiteTensorRTBackend,
    )

    __all__.append("SSDLiteTensorRTBackend")
except ImportError:
    pass
