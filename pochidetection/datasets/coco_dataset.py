"""COCO形式の物体検出データセット."""

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import RTDetrImageProcessor

from pochidetection.interfaces.dataset import IDetectionDataset
from pochidetection.utils.category_utils import (
    build_category_id_to_idx,
    filter_categories,
)


class CocoDetectionDataset(Dataset[dict[str, Any]], IDetectionDataset):
    """COCO形式の物体検出データセット.

    COCO形式のディレクトリ構造:
        root/
        ├── image1.jpg
        ├── image2.jpg
        └── annotations.json         # COCO形式アノテーション
            または
        └── instances_train2017.json # COCO形式アノテーション

    アノテーションJSON形式:
        {
            "images": [{"id": 1, "file_name": "...", "width": ..., "height": ...}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [...]}],
            "categories": [{"id": 1, "name": "..."}]
        }

    Attributes:
        _root: データセットのルートディレクトリ.
        _processor: RTDetrImageProcessor.
        _annotation_file: アノテーションファイルのパス.
        _images: 画像情報のリスト.
        _annotations: アノテーション情報 (image_idでグループ化).
        _categories: カテゴリ情報のリスト (背景クラスを除く).
        _category_id_to_idx: カテゴリIDから連続インデックスへのマッピング.
    """

    def __init__(
        self,
        root: str | Path,
        processor: RTDetrImageProcessor,
        annotation_file: str | None = None,
    ) -> None:
        """CocoDetectionDatasetを初期化.

        Args:
            root: データセットのルートディレクトリパス.
            processor: RTDetrImageProcessor.
            annotation_file: アノテーションファイル名.
                指定しない場合, annotations.json または instances_*.json を自動検索.

        Raises:
            FileNotFoundError: アノテーションファイルが見つからない場合.
        """
        self._root = Path(root)
        self._processor = processor

        # アノテーションファイルを探す
        self._annotation_file = self._find_annotation_file(annotation_file)

        # アノテーションを読み込み
        self._images, self._annotations, self._categories = self._load_annotations()

        # カテゴリIDを連続インデックスにマッピング
        self._category_id_to_idx = build_category_id_to_idx(self._categories)

    def _find_annotation_file(self, annotation_file: str | None) -> Path:
        """アノテーションファイルを探す.

        Args:
            annotation_file: 指定されたアノテーションファイル名.

        Returns:
            アノテーションファイルのパス.

        Raises:
            FileNotFoundError: アノテーションファイルが見つからない場合.
        """
        if annotation_file:
            path = self._root / annotation_file
            if path.exists():
                return path
            raise FileNotFoundError(f"アノテーションファイルが見つかりません: {path}")

        # 自動検索: annotations.json
        candidates = [
            self._root / "annotations.json",
            *list(self._root.glob("instances_*.json")),
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"アノテーションファイルが見つかりません. "
            f"検索パス: {self._root}/annotations.json または instances_*.json"
        )

    def _load_annotations(
        self,
    ) -> tuple[
        list[dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]
    ]:
        """アノテーションファイルを読み込む.

        Returns:
            (images, annotations_by_image_id, categories) のタプル.
        """
        with open(self._annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        # 背景クラスを除外し, カテゴリIDの昇順でソート.
        # JSON 内の出現順に依存しない一意のマッピングを保証する.
        categories = filter_categories(data.get("categories", []))

        # image_idでアノテーションをグループ化
        annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
        for ann in annotations:
            image_id = ann["image_id"]
            if image_id not in annotations_by_image_id:
                annotations_by_image_id[image_id] = []
            annotations_by_image_id[image_id].append(ann)

        return images, annotations_by_image_id, categories

    def __len__(self) -> int:
        """データセット内のサンプル数を返す.

        Returns:
            サンプル数.
        """
        return len(self._images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """インデックスでサンプルを取得.

        Args:
            idx: サンプルのインデックス.

        Returns:
            以下のキーを含む辞書:
            - pixel_values: 前処理済み画像テンソル (C, H, W)
            - labels: RT-DETR形式のターゲット {"boxes": (N, 4), "class_labels": (N,)}
        """
        image_info = self._images[idx]
        image_id = image_info["id"]

        # 画像を読み込み
        image_path = self._root / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        # アノテーションを取得
        annotations = self._annotations.get(image_id, [])

        # ボックスとラベルを抽出 (正規化cxcywh形式)
        boxes = []
        labels = []
        for ann in annotations:
            cat_id = ann["category_id"]
            if cat_id not in self._category_id_to_idx:
                continue  # 背景クラスはスキップ

            # COCO形式: [x, y, w, h] -> 正規化 [cx, cy, w, h]
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue  # ゼロサイズの bbox はスキップ
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            boxes.append([cx, cy, nw, nh])
            labels.append(self._category_id_to_idx[cat_id])

        # 前処理
        encoding = self._processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        # ターゲット (RT-DETRの形式)
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

    def get_categories(self) -> list[dict[str, Any]]:
        """カテゴリ情報を取得.

        Returns:
            カテゴリ情報のリスト. 各要素は {"id": int, "name": str} の形式.
        """
        return self._categories

    def get_num_classes(self) -> int:
        """クラス数を取得.

        Returns:
            クラス数.
        """
        return len(self._categories)

    def get_category_names(self) -> list[str]:
        """カテゴリ名のリストを取得.

        Returns:
            カテゴリ名のリスト (連続インデックス順).
        """
        return [cat["name"] for cat in self._categories]
