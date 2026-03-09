"""推論パイプラインモジュール."""

from pochidetection.inference.onnx_backend import OnnxBackend
from pochidetection.inference.pytorch_backend import PyTorchBackend
from pochidetection.inference.ssdlite_onnx_backend import SSDLiteOnnxBackend
from pochidetection.inference.ssdlite_pytorch_backend import SSDLitePyTorchBackend

__all__ = [
    "OnnxBackend",
    "PyTorchBackend",
    "SSDLiteOnnxBackend",
    "SSDLitePyTorchBackend",
]

try:
    from pochidetection.inference.tensorrt_backend import TensorRTBackend

    __all__.append("TensorRTBackend")
except ImportError:
    pass
