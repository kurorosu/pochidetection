"""Pochidetection - pochitrainの設計思想に基づいた物体検出フレームワーク."""

__version__ = "0.2.0"

# Core
from pochidetection.core import DetectionCollator

# Concrete implementations
from pochidetection.datasets import CocoDetectionDataset

# Interfaces
from pochidetection.interfaces import (
    IDetectionDataset,
    IDetectionModel,
)

# Logging
from pochidetection.logging import LoggerManager, LogLevel
from pochidetection.models import RTDetrModel

# Utils
from pochidetection.utils import ConfigLoader, WorkspaceManager

__all__ = [
    # Interfaces
    "IDetectionModel",
    "IDetectionDataset",
    # Logging
    "LoggerManager",
    "LogLevel",
    # Concrete implementations
    "RTDetrModel",
    "CocoDetectionDataset",
    # Core
    "DetectionCollator",
    # Utils
    "ConfigLoader",
    "WorkspaceManager",
]
