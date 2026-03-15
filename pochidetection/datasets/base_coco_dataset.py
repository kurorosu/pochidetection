"""COCO 形式データセットの基底クラス."""

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from pochidetection.interfaces.dataset import DatasetSampleDict, IDetectionDataset
from pochidetection.utils.category_utils import (
    build_category_id_to_idx,
    filter_categories,
)


class BaseCocoDataset(Dataset[DatasetSampleDict], IDetectionDataset):
    """COCO 形式データセットの基底クラス.

    アノテーション読み込み, カテゴリ管理, 画像ロードの共通ロジックを提供する.
    サブクラスは _transform_sample を実装して前処理と座標変換を定義する.

    Attributes:
        _root: データセットのルートディレクトリ.
        _annotation_file: アノテーションファイルのパス.
        _images: 画像情報のリスト.
        _annotations: アノテーション情報 (image_id でグループ化).
        _categories: カテゴリ情報のリスト (背景クラスを除く).
        _category_id_to_idx: カテゴリ ID から連続インデックスへのマッピング.
    """

    def __init__(
        self,
        root: str | Path,
        annotation_file: str | None = None,
    ) -> None:
        """初期化.

        Args:
            root: データセットのルートディレクトリパス.
            annotation_file: アノテーションファイル名.
                指定しない場合, annotations.json または instances_*.json を自動検索.

        Raises:
            FileNotFoundError: アノテーションファイルが見つからない場合.
        """
        self._root = Path(root)
        self._annotation_file = self._find_annotation_file(annotation_file)
        images, annotations_by_id, self._categories = self._load_annotations()
        self._category_id_to_idx = build_category_id_to_idx(self._categories)
        self._images = images
        # __getitem__() 毎回のフィルタリングを避けるため, 初期化時に一括実行する.
        self._annotations = self._filter_annotations(annotations_by_id)

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

        # 大規模 JSON (数百 MB) でのメモリ常駐を防ぐため,
        # 必要フィールド抽出後に元データを即座に解放する.
        images: list[dict[str, Any]] = data.get("images", [])
        raw_annotations: list[dict[str, Any]] = data.get("annotations", [])
        raw_categories: list[dict[str, Any]] = data.get("categories", [])
        del data

        # 背景クラスを除外し, カテゴリ ID の昇順でソート.
        # JSON 内の出現順に依存しない一意のマッピングを保証する.
        categories = filter_categories(raw_categories)
        del raw_categories

        # image_id でアノテーションをグループ化
        annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
        for ann in raw_annotations:
            image_id = ann["image_id"]
            if image_id not in annotations_by_image_id:
                annotations_by_image_id[image_id] = []
            annotations_by_image_id[image_id].append(ann)
        del raw_annotations

        return images, annotations_by_image_id, categories

    def _filter_annotations(
        self,
        annotations_by_id: dict[int, list[dict[str, Any]]],
    ) -> dict[int, list[dict[str, Any]]]:
        """無効なアノテーションを除外する.

        カテゴリ ID が未知, または bbox のサイズがゼロ以下のものを除外する.

        Args:
            annotations_by_id: image_id でグループ化されたアノテーション.

        Returns:
            フィルタ済みアノテーション (image_id でグループ化).
        """
        filtered: dict[int, list[dict[str, Any]]] = {}
        for image_id, anns in annotations_by_id.items():
            valid = [
                ann
                for ann in anns
                if ann["category_id"] in self._category_id_to_idx
                and ann["bbox"][2] > 0
                and ann["bbox"][3] > 0
            ]
            if valid:
                filtered[image_id] = valid
        return filtered

    def __len__(self) -> int:
        """データセット内のサンプル数を返す.

        Returns:
            サンプル数.
        """
        return len(self._images)

    def __getitem__(self, idx: int) -> DatasetSampleDict:
        """インデックスでサンプルを取得.

        Args:
            idx: サンプルのインデックス.

        Returns:
            以下のキーを含む辞書:
            - pixel_values: 前処理済み画像テンソル (C, H, W)
            - labels: ターゲット辞書 (形式はサブクラスに依存)
        """
        image_info = self._images[idx]
        image_id = image_info["id"]

        image_path = self._root / image_info["file_name"]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        orig_w, orig_h = image.size

        annotations = self._annotations.get(image_id, [])

        return self._transform_sample(image, annotations, orig_w, orig_h)

    @abstractmethod
    def _transform_sample(
        self,
        image: Image.Image,
        annotations: list[dict[str, Any]],
        orig_w: int,
        orig_h: int,
    ) -> DatasetSampleDict:
        """画像とアノテーションを変換してサンプルを生成する.

        Args:
            image: PIL 画像.
            annotations: 有効なアノテーションのリスト (ゼロサイズ除外済み).
            orig_w: 元画像の幅.
            orig_h: 元画像の高さ.

        Returns:
            pixel_values と labels を含む辞書.
        """
        pass

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
