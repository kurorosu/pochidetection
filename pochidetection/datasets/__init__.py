"""物体検出データセットパッケージ."""

from pochidetection.datasets.coco_dataset import CocoDetectionDataset
from pochidetection.datasets.ssd_coco_dataset import SsdCocoDataset

__all__ = ["CocoDetectionDataset", "SsdCocoDataset"]
