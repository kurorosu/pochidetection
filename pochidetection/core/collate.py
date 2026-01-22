"""物体検出用Collate関数.

DataLoaderのバッチ作成時に使用するcollate関数を提供する.
RT-DETRのImageProcessorを使用して画像を前処理する.
"""

from typing import Any

import torch
from PIL import Image
from transformers import RTDetrImageProcessor


class DetectionCollator:
    """物体検出用Collator.

    DataLoaderのcollate_fnとして使用し, バッチを作成する.
    RT-DETRのImageProcessorを使用して画像を前処理する.

    Attributes:
        _processor: RTDetrImageProcessor.
        _image_size: 画像サイズ.
    """

    def __init__(
        self,
        model_name: str = "PekingU/rtdetr_r50vd",
        image_size: int = 640,
    ) -> None:
        """DetectionCollatorを初期化.

        Args:
            model_name: HuggingFaceモデル名 (ImageProcessor取得用).
            image_size: 画像サイズ.
        """
        self._processor = RTDetrImageProcessor.from_pretrained(model_name)
        self._image_size = image_size

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """バッチをcollate.

        Args:
            batch: データセットからのサンプルリスト.
                各サンプルは以下のキーを含む:
                - image: PIL.Image または torch.Tensor
                - boxes: バウンディングボックス (N, 4) [x_min, y_min, x_max, y_max]
                - labels: クラスラベル (N,)
                - image_id: 画像ID
                - orig_size: 元の画像サイズ (height, width)

        Returns:
            以下のキーを含む辞書:
            - pixel_values: 前処理済み画像テンソル (B, C, H, W)
            - boxes: ボックスのリスト (正規化済み, cxcywh形式)
            - labels: ラベルのリスト
            - image_ids: 画像IDのリスト
            - orig_sizes: 元サイズのリスト
        """
        images = []
        all_boxes = []
        all_labels = []
        image_ids = []
        orig_sizes = []

        for sample in batch:
            # 画像を取得 (PIL.Imageに変換)
            image = sample["image"]
            if isinstance(image, torch.Tensor):
                # テンソルからPIL.Imageに変換
                if image.dim() == 3 and image.shape[0] == 3:
                    # (C, H, W) -> (H, W, C)
                    image = image.permute(1, 2, 0)
                image = Image.fromarray(
                    (image.numpy() * 255).astype("uint8")
                    if image.max() <= 1.0
                    else image.numpy().astype("uint8")
                )
            images.append(image)

            # ボックスとラベルを取得
            boxes = sample["boxes"]  # (N, 4) [x_min, y_min, x_max, y_max]
            labels = sample["labels"]  # (N,)

            # ボックスを正規化 (xyxy -> cxcywh, 画像サイズで正規化)
            orig_h, orig_w = sample["orig_size"]
            if len(boxes) > 0:
                # 正規化 (0-1)
                normalized_boxes = boxes.clone().float()
                normalized_boxes[:, [0, 2]] /= orig_w
                normalized_boxes[:, [1, 3]] /= orig_h

                # xyxy -> cxcywh
                x_min, y_min, x_max, y_max = normalized_boxes.unbind(-1)
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                w = x_max - x_min
                h = y_max - y_min
                cxcywh_boxes = torch.stack([cx, cy, w, h], dim=-1)
            else:
                cxcywh_boxes = torch.zeros((0, 4), dtype=torch.float32)

            all_boxes.append(cxcywh_boxes)
            all_labels.append(labels)
            image_ids.append(sample["image_id"])
            orig_sizes.append(sample["orig_size"])

        # ImageProcessorで画像を前処理
        processed = self._processor(
            images=images,
            return_tensors="pt",
        )

        return {
            "pixel_values": processed["pixel_values"],
            "boxes": all_boxes,
            "labels": all_labels,
            "image_ids": image_ids,
            "orig_sizes": orig_sizes,
        }
