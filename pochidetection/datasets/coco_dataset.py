"""COCO形式の物体検出データセット (RT-DETR 用)."""

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import v2
from transformers import RTDetrImageProcessor

from pochidetection.core.letterbox import apply_letterbox, compute_letterbox_params
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
        _letterbox: letterbox リサイズを適用するかどうか.
        _image_size: letterbox 有効時の target サイズ (H, W). None なら processor
            の内部 resize (単純リサイズ) に委ねる.
    """

    def __init__(
        self,
        root: str | Path,
        processor: RTDetrImageProcessor,
        annotation_file: str | None = None,
        augmentation: v2.Compose | None = None,
        letterbox: bool = True,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        """CocoDetectionDatasetを初期化.

        Args:
            root: データセットのルートディレクトリパス.
            processor: RTDetrImageProcessor.
            annotation_file: アノテーションファイル名.
                指定しない場合, annotations.json または instances_*.json を自動検索.
            augmentation: 学習時に適用する augmentation パイプライン.
            letterbox: True (既定) で dataset 側で letterbox を適用し, processor は
                normalize のみ (do_resize=False) 委譲する. False で従来経路 (processor
                の内部 resize) に戻る.
            image_size: letterbox 有効時の target サイズ (height, width). None の場合は
                ``processor.size`` から自動解決する.

        Raises:
            FileNotFoundError: アノテーションファイルが見つからない場合.
            ValueError: letterbox=True で image_size 解決に失敗した場合.
        """
        self._processor = processor
        self._letterbox = letterbox
        self._image_size: tuple[int, int] | None = None
        if letterbox:
            self._image_size = image_size or self._resolve_processor_size(processor)
        super().__init__(root, annotation_file, augmentation)

    @staticmethod
    def _resolve_processor_size(
        processor: RTDetrImageProcessor,
    ) -> tuple[int, int]:
        """Extract target (H, W) from the HF image processor's ``size`` attribute.

        Args:
            processor: HF image processor (RTDetr).

        Returns:
            target (height, width).

        Raises:
            ValueError: ``processor.size`` から (H, W) を決定できない場合.
        """
        size = getattr(processor, "size", None)
        if isinstance(size, dict):
            # HF の convention: {"height": H, "width": W} or {"shortest_edge": ...}
            if "height" in size and "width" in size:
                return (int(size["height"]), int(size["width"]))
            if "shortest_edge" in size and "longest_edge" in size:
                return (int(size["shortest_edge"]), int(size["longest_edge"]))
        raise ValueError(
            "letterbox=True で image_size が未指定の場合, processor.size から "
            "(height, width) を決定できる必要があります. 明示的に image_size を "
            f"渡してください. processor.size={size}"
        )

    def _transform_sample(
        self,
        image: Image.Image,
        annotations: list[dict[str, Any]],
        orig_w: int,
        orig_h: int,
    ) -> DatasetSampleDict:
        """画像とアノテーションを RT-DETR 形式に変換.

        letterbox 有効時は dataset 側で letterbox (アスペクト比維持 + padding) を
        適用した上で, processor には normalize のみ委譲 (do_resize=False). bbox は
        letterbox 後の pixel 座標を target サイズ基準で cxcywh 正規化する.

        letterbox 無効時は従来経路 (processor の内部 resize + 元画像基準の正規化).

        Args:
            image: PIL 画像.
            annotations: 有効なアノテーションのリスト.
            orig_w: 元画像の幅.
            orig_h: 元画像の高さ.

        Returns:
            pixel_values と labels (boxes: cxcywh 正規化, class_labels: 0-indexed) を含む辞書.
        """
        if self._letterbox and self._image_size is not None:
            params = compute_letterbox_params((orig_h, orig_w), self._image_size)
            processed_image = apply_letterbox(image, params, pad_value=0)
            target_h, target_w = self._image_size
            boxes = []
            labels = []
            for ann in annotations:
                x, y, w, h = ann["bbox"]
                # 元画像 → letterbox 後 pixel 座標
                new_x = x * params.scale + params.pad_left
                new_y = y * params.scale + params.pad_top
                new_w = w * params.scale
                new_h = h * params.scale
                # target サイズ基準で cxcywh 正規化
                cx = (new_x + new_w / 2) / target_w
                cy = (new_y + new_h / 2) / target_h
                nw = new_w / target_w
                nh = new_h / target_h
                boxes.append([cx, cy, nw, nh])
                labels.append(self._category_id_to_idx[ann["category_id"]])

            encoding = self._processor(
                images=processed_image, do_resize=False, return_tensors="pt"
            )
        else:
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
