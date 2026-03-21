"""COCO形式の物体検出データセット (RT-DETR 用)."""

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import v2
from transformers import RTDetrImageProcessor

from pochidetection.datasets.base_coco_dataset import BaseCocoDataset
from pochidetection.interfaces.dataset import DatasetSampleDict


class CocoDetectionDataset(BaseCocoDataset):
    """COCO形式の物体検出データセット.

    RT-DETR モデル用に, 正規化 cxcywh 座標と 0-indexed ラベルを返す.

    COCO形式のディレクトリ構造:
        root/
        ├── image1.jpg
        ├── image2.jpg
        └── annotations.json         # COCO形式アノテーション
            または
        └── instances_train2017.json # COCO形式アノテーション

    Attributes:
        _processor: RTDetrImageProcessor.
    """

    def __init__(
        self,
        root: str | Path,
        processor: RTDetrImageProcessor,
        annotation_file: str | None = None,
        augmentation: v2.Compose | None = None,
    ) -> None:
        """CocoDetectionDatasetを初期化.

        Args:
            root: データセットのルートディレクトリパス.
            processor: RTDetrImageProcessor.
            annotation_file: アノテーションファイル名.
                指定しない場合, annotations.json または instances_*.json を自動検索.
            augmentation: 学習時に適用する augmentation パイプライン.

        Raises:
            FileNotFoundError: アノテーションファイルが見つからない場合.
        """
        self._processor = processor
        super().__init__(root, annotation_file, augmentation)

    def _transform_sample(
        self,
        image: Image.Image,
        annotations: list[dict[str, Any]],
        orig_w: int,
        orig_h: int,
    ) -> DatasetSampleDict:
        """画像とアノテーションを RT-DETR 形式に変換.

        Args:
            image: PIL 画像.
            annotations: 有効なアノテーションのリスト.
            orig_w: 元画像の幅.
            orig_h: 元画像の高さ.

        Returns:
            pixel_values と labels (boxes: cxcywh 正規化, class_labels: 0-indexed) を含む辞書.
        """
        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            boxes.append([cx, cy, nw, nh])
            labels.append(self._category_id_to_idx[ann["category_id"]])

        encoding = self._processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        target = {
            "boxes": (
                torch.tensor(boxes, dtype=torch.float32)
                if boxes
                else torch.zeros((0, 4))
            ),
            "class_labels": (
                torch.tensor(labels, dtype=torch.int64)
                if labels
                else torch.zeros((0,), dtype=torch.int64)
            ),
        }

        return {
            "pixel_values": pixel_values,
            "labels": target,
        }
