"""Pochidetection - pochitrainの設計思想に基づいた物体検出フレームワーク."""

__version__ = "0.12.0"

from pochidetection.core import DetectionCollator
from pochidetection.datasets import CocoDetectionDataset, SsdCocoDataset
from pochidetection.interfaces import (
    IDetectionDataset,
    IDetectionModel,
)
from pochidetection.logging import LoggerManager, LogLevel
from pochidetection.models import RTDetrModel, SSD300Model, SSDLiteModel
from pochidetection.utils import ConfigLoader, WorkspaceManager

__all__ = [
    "ConfigLoader",
    "CocoDetectionDataset",
    "DetectionCollator",
    "IDetectionDataset",
    "IDetectionModel",
    "LogLevel",
    "LoggerManager",
    "RTDetrModel",
    "SSD300Model",
    "SSDLiteModel",
    "SsdCocoDataset",
    "WorkspaceManager",
]
