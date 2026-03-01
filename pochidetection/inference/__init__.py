"""推論パイプラインモジュール."""

from pochidetection.inference.onnx_backend import OnnxBackend
from pochidetection.inference.pytorch_backend import PyTorchBackend

__all__ = [
    "OnnxBackend",
    "PyTorchBackend",
]
