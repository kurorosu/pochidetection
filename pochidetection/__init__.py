"""Pochidetection - pochitrainの設計思想に基づいた物体検出フレームワーク."""

# Factory
# Concrete implementations
from pochidetection.datasets import CocoDetectionDataset
from pochidetection.factories import ComponentFactory

# Interfaces
from pochidetection.interfaces import (
    IDetectionDataset,
    IDetectionLoss,
    IDetectionMetrics,
    IDetectionModel,
)

# Logging
from pochidetection.logging import LoggerManager, LogLevel
from pochidetection.losses import DetectionLoss
from pochidetection.metrics import DetectionMetrics
from pochidetection.models import RTDetrModel

# Utils
from pochidetection.utils import ConfigLoader

# コンポーネント登録 (Factory + Registry パターン)
ComponentFactory.register_model("RTDetr", RTDetrModel)
ComponentFactory.register_loss("DetectionLoss", DetectionLoss)
ComponentFactory.register_metrics("DetectionMetrics", DetectionMetrics)
ComponentFactory.register_dataset("CocoDetectionDataset", CocoDetectionDataset)

__all__ = [
    # Factory
    "ComponentFactory",
    # Interfaces
    "IDetectionModel",
    "IDetectionLoss",
    "IDetectionMetrics",
    "IDetectionDataset",
    # Logging
    "LoggerManager",
    "LogLevel",
    # Concrete implementations
    "RTDetrModel",
    "DetectionLoss",
    "DetectionMetrics",
    "CocoDetectionDataset",
    # Utils
    "ConfigLoader",
]
