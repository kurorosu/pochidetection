"""物体検出コンポーネントのインターフェース (DIP - 依存性逆転の原則)."""

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
