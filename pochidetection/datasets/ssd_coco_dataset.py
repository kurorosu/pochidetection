"""SSDLite 用 COCO 形式の物体検出データセット."""

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from pochidetection.datasets.base_coco_dataset import BaseCocoDataset


class SsdCocoDataset(BaseCocoDataset):
    """SSDLite 用 COCO 形式の物体検出データセット.

    ラベルは 0-indexed で返す.
    背景クラスオフセットは SSDLiteModel 側で管理する.
    ボックスは xyxy ピクセル座標で返す.

    Attributes:
        _image_size: リサイズ後の画像サイズ (height, width).
        _transform: 画像前処理の transforms.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: dict[str, int],
        annotation_file: str | None = None,
    ) -> None:
        """初期化.

        Args:
            root: データセットのルートディレクトリパス.
            image_size: リサイズ先の画像サイズ {"height": int, "width": int}.
            annotation_file: アノテーションファイル名.
                指定しない場合, annotations.json または instances_*.json を自動検索.

        Raises:
            FileNotFoundError: アノテーションファイルが見つからない場合.
        """
        self._image_size = (image_size["height"], image_size["width"])
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        super().__init__(root, annotation_file)

    def _transform_sample(
        self,
        image: Image.Image,
        annotations: list[dict[str, Any]],
        orig_w: int,
        orig_h: int,
    ) -> dict[str, Any]:
        """画像とアノテーションを SSD 形式に変換.

        Args:
            image: PIL 画像.
            annotations: 有効なアノテーションのリスト.
            orig_w: 元画像の幅.
            orig_h: 元画像の高さ.

        Returns:
            pixel_values と labels (boxes: xyxy, class_labels: 0-indexed) を含む辞書.
        """
        target_h, target_w = self._image_size
        image_resized = image.resize((target_w, target_h))

        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            boxes.append([x1, y1, x2, y2])
            labels.append(self._category_id_to_idx[ann["category_id"]])

        pixel_values = self._transform(image_resized)

        target = {
            "boxes": (
                torch.tensor(boxes, dtype=torch.float32)
                if boxes
                else torch.zeros((0, 4), dtype=torch.float32)
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
