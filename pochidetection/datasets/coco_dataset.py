"""COCO形式の物体検出データセット."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from pochidetection.interfaces.dataset import IDetectionDataset


class CocoDetectionDataset(Dataset[dict[str, Any]], IDetectionDataset):
    """COCO形式の物体検出データセット.

    COCO形式のディレクトリ構造:
        root/
        ├── JPEGImages/              # 元画像
        │   ├── image1.jpg
        │   └── image2.jpg
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
        _annotation_file: アノテーションファイルのパス.
        _transform: 画像に適用するtransform.
        _images: 画像情報のリスト.
        _annotations: アノテーション情報 (image_idでグループ化).
        _categories: カテゴリ情報のリスト.
        _category_id_to_idx: カテゴリIDから連続インデックスへのマッピング.
    """

    def __init__(
        self,
        root: str | Path,
        annotation_file: str | None = None,
        transform: Callable[..., Any] | None = None,
    ) -> None:
        """CocoDetectionDatasetを初期化.

        Args:
            root: データセットのルートディレクトリパス.
            annotation_file: アノテーションファイル名.
                指定しない場合, annotations.json または instances_*.json を自動検索.
            transform: 画像に適用するtransform.

        Raises:
            FileNotFoundError: アノテーションファイルが見つからない場合.
        """
        self._root = Path(root)
        self._transform = transform

        # アノテーションファイルを探す
        self._annotation_file = self._find_annotation_file(annotation_file)

        # アノテーションを読み込み
        self._images, self._annotations, self._categories = self._load_annotations()

        # カテゴリIDを連続インデックスにマッピング
        self._category_id_to_idx = {
            cat["id"]: idx for idx, cat in enumerate(self._categories)
        }

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
        categories = data.get("categories", [])

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
            - image: 画像テンソル (C, H, W)
            - boxes: バウンディングボックス (N, 4) [x_min, y_min, x_max, y_max]
            - labels: クラスラベル (N,)
            - image_id: 画像ID
            - orig_size: 元の画像サイズ (height, width)
        """
        image_info = self._images[idx]
        image_id = image_info["id"]

        # 画像を読み込み
        image_path = self._root / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        orig_size = (image_info["height"], image_info["width"])

        # アノテーションを取得
        annotations = self._annotations.get(image_id, [])

        # ボックスとラベルを抽出
        boxes = []
        labels = []
        for ann in annotations:
            # COCO形式: [x, y, width, height] -> [x_min, y_min, x_max, y_max]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            # カテゴリIDを連続インデックスに変換
            labels.append(self._category_id_to_idx[ann["category_id"]])

        # テンソルに変換
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        # transformを適用
        if self._transform:
            image = self._transform(image)

        return {
            "image": image,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": image_id,
            "orig_size": orig_size,
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
