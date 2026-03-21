"""SSDLite 用 COCO 形式の物体検出データセット."""

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from pochidetection.configs.schemas import ImageSizeDict
from pochidetection.datasets.base_coco_dataset import BaseCocoDataset
from pochidetection.interfaces.dataset import DatasetSampleDict


class SsdCocoDataset(BaseCocoDataset):
    """SSDLite 用 COCO 形式の物体検出データセット.

    ラベルは 0-indexed で返す.
    背景クラスオフセットは SSDLiteModel 側で管理する.
    ボックスは xyxy ピクセル座標で返す.

    torchvision.transforms.v2 を使用し, 画像リサイズとボックス座標変換を
    一括で処理する.

    Attributes:
        _image_size: リサイズ後の画像サイズ (height, width).
        _transform: 画像・ボックスを一括処理する v2 transforms パイプライン.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: ImageSizeDict,
        annotation_file: str | None = None,
        augmentation: v2.Compose | None = None,
    ) -> None:
        """初期化.

        Args:
            root: データセットのルートディレクトリパス.
            image_size: リサイズ先の画像サイズ {"height": int, "width": int}.
            annotation_file: アノテーションファイル名.
                指定しない場合, annotations.json または instances_*.json を自動検索.
            augmentation: 学習時に適用する augmentation パイプライン.

        Raises:
            FileNotFoundError: アノテーションファイルが見つからない場合.
        """
        self._image_size = (image_size["height"], image_size["width"])
        self._transform = v2.Compose(
            [
                v2.Resize(self._image_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        super().__init__(root, annotation_file, augmentation)

    def _transform_sample(
        self,
        image: Image.Image,
        annotations: list[dict[str, Any]],
        orig_w: int,
        orig_h: int,
    ) -> DatasetSampleDict:
        """画像とアノテーションを SSD 形式に変換.

        v2 transforms が画像リサイズとボックス座標変換を一括処理する.

        Args:
            image: PIL 画像.
            annotations: 有効なアノテーションのリスト.
            orig_w: 元画像の幅.
            orig_h: 元画像の高さ.

        Returns:
            pixel_values と labels (boxes: xyxy, class_labels: 0-indexed) を含む辞書.
        """
        raw_boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            raw_boxes.append([x, y, x + w, y + h])
            labels.append(self._category_id_to_idx[ann["category_id"]])

        if raw_boxes:
            boxes_tensor = tv_tensors.BoundingBoxes(
                raw_boxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(orig_h, orig_w),
            )
        else:
            boxes_tensor = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(orig_h, orig_w),
            )

        pixel_values, transformed_boxes = self._transform(image, boxes_tensor)

        target = {
            "boxes": torch.as_tensor(transformed_boxes, dtype=torch.float32),
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
