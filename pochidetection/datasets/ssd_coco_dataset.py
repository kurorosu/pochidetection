"""SSDLite 用 COCO 形式の物体検出データセット."""

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from pochidetection.interfaces.dataset import IDetectionDataset
from pochidetection.utils.category_utils import (
    build_category_id_to_idx,
    filter_categories,
)


class SsdCocoDataset(Dataset[dict[str, Any]], IDetectionDataset):
    """SSDLite 用 COCO 形式の物体検出データセット.

    SSD モデルは背景クラス (label=0) を使用するため,
    ラベルは 1-indexed (背景=0) で返す.
    ボックスは xyxy ピクセル座標で返す.

    Attributes:
        _root: データセットのルートディレクトリ.
        _image_size: リサイズ後の画像サイズ (height, width).
        _transform: 画像前処理の transforms.
        _images: 画像情報のリスト.
        _annotations: アノテーション情報 (image_id でグループ化).
        _categories: カテゴリ情報のリスト (背景クラスを除く).
        _category_id_to_idx: カテゴリ ID から連続インデックスへのマッピング.
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
        self._root = Path(root)
        self._image_size = (image_size["height"], image_size["width"])

        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self._annotation_file = self._find_annotation_file(annotation_file)
        self._images, self._annotations, self._categories = self._load_annotations()
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
        categories = filter_categories(data.get("categories", []))

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
            - labels: SSD 形式のターゲット
                {"boxes": (N, 4) xyxy, "labels": (N,) 1-indexed}
        """
        image_info = self._images[idx]
        image_id = image_info["id"]

        image_path = self._root / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        target_h, target_w = self._image_size
        image_resized = image.resize((target_w, target_h))

        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        annotations = self._annotations.get(image_id, [])
        boxes = []
        labels = []
        for ann in annotations:
            cat_id = ann["category_id"]
            if cat_id not in self._category_id_to_idx:
                continue

            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            boxes.append([x1, y1, x2, y2])

            # 1-indexed (背景=0)
            labels.append(self._category_id_to_idx[cat_id] + 1)
        pixel_values = self._transform(image_resized)

        target = {
            "boxes": (
                torch.tensor(boxes, dtype=torch.float32)
                if boxes
                else torch.zeros((0, 4), dtype=torch.float32)
            ),
            "labels": (
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
