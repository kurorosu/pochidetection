"""コアモジュール."""

from pochidetection.core.collate import DetectionCollator
from pochidetection.core.detection import Detection, OutputWrapper

__all__ = ["DetectionCollator", "Detection", "OutputWrapper"]
