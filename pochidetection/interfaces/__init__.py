"""物体検出コンポーネントのインターフェース群を提供."""

from pochidetection.interfaces.dataset import IDetectionDataset
from pochidetection.interfaces.loss import IDetectionLoss
from pochidetection.interfaces.metrics import IDetectionMetrics
from pochidetection.interfaces.model import IDetectionModel

__all__ = [
    "IDetectionModel",
    "IDetectionLoss",
    "IDetectionMetrics",
    "IDetectionDataset",
]
