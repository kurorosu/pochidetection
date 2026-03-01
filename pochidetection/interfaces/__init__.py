"""物体検出コンポーネントのインターフェース群を提供."""

from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.interfaces.dataset import IDetectionDataset
from pochidetection.interfaces.model import IDetectionModel
from pochidetection.interfaces.plotter import IPlotter

__all__ = [
    "IDetectionDataset",
    "IDetectionModel",
    "IInferenceBackend",
    "IPlotter",
]
