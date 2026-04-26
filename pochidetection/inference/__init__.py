"""推論バックエンドモジュール."""

from pochidetection.inference.builder import build_pytorch_backend
from pochidetection.inference.rtdetr.onnx_backend import RTDetrOnnxBackend
from pochidetection.inference.rtdetr.pytorch_backend import RTDetrPyTorchBackend
from pochidetection.inference.ssd.pytorch_backend import SsdPyTorchBackend
from pochidetection.inference.ssdlite.onnx_backend import SSDLiteOnnxBackend

__all__ = [
    "RTDetrOnnxBackend",
    "RTDetrPyTorchBackend",
    "SSDLiteOnnxBackend",
    "SsdPyTorchBackend",
    "build_pytorch_backend",
]

try:
    from pochidetection.inference.rtdetr.tensorrt_backend import RTDetrTensorRTBackend

    __all__.append("RTDetrTensorRTBackend")
except ImportError:
    pass

try:
    from pochidetection.inference.ssdlite.tensorrt_backend import (
        SSDLiteTensorRTBackend,
    )

    __all__.append("SSDLiteTensorRTBackend")
except ImportError:
    pass
