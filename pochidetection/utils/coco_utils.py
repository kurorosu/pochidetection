"""COCO アノテーション操作の共通ユーティリティ."""

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from pochidetection.logging import LoggerManager
from pochidetection.utils.category_utils import (
    build_category_id_to_idx,
    filter_categories,
)

logger = LoggerManager().get_logger(__name__)


def extract_basename(file_name: str) -> str:
    r"""パス区切り文字を考慮してベースネームを抽出する.

    COCO アノテーションの file_name は OS に依存して "/" や "\\" を含む場合がある.
    両方の区切り文字を考慮してベースネームを返す.

    Args:
        file_name: アノテーション内の file_name 文字列.

    Returns:
        ベースネーム (ファイル名のみ).
    """
    if "\\" in file_name:
        return PureWindowsPath(file_name).name
    return PurePosixPath(file_name).name


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    """COCO の [x, y, w, h] を [x1, y1, x2, y2] に変換する.

    Args:
        bbox: COCO フォーマットの [x, y, w, h].

    Returns:
        [x1, y1, x2, y2] フォーマット.
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


class CocoGroundTruth:
    """COCO アノテーションの GT 読み込み結果.

    Attributes:
        image_id_by_filename: ファイル名から image_id へのマッピング.
        filenames_by_image_id: image_id からファイル名リストへの逆引きマッピング.
        category_id_to_idx: カテゴリ ID から連続インデックスへのマッピング.
        categories: フィルタ済みカテゴリリスト.
        gt_by_image_id: image_id ごとの GT アノテーションリスト.
    """

    def __init__(
        self,
        image_id_by_filename: dict[str, int],
        filenames_by_image_id: dict[int, list[str]],
        category_id_to_idx: dict[int, int],
        categories: list[dict[str, Any]],
        gt_by_image_id: dict[int, list[dict[str, Any]]],
    ) -> None:
        """初期化.

        Args:
            image_id_by_filename: ファイル名から image_id へのマッピング.
            filenames_by_image_id: image_id からファイル名リストへの逆引き.
            category_id_to_idx: カテゴリ ID → 連続インデックス.
            categories: フィルタ済みカテゴリリスト.
            gt_by_image_id: image_id ごとの GT アノテーションリスト.
        """
        self.image_id_by_filename = image_id_by_filename
        self.filenames_by_image_id = filenames_by_image_id
        self.category_id_to_idx = category_id_to_idx
        self.categories = categories
        self.gt_by_image_id = gt_by_image_id

    def gt_by_filename(self) -> dict[str, list[dict[str, Any]]]:
        """ファイル名をキーとした GT アノテーション辞書を構築する.

        Returns:
            ファイル名をキー, GT アノテーションリストを値とする辞書.
        """
        result: dict[str, list[dict[str, Any]]] = {}
        for image_id, anns in self.gt_by_image_id.items():
            filenames = self.filenames_by_image_id.get(image_id, [])
            if filenames:
                # 最短のファイル名 (basename があればそちら) をキーにする
                key = min(filenames, key=len)
                result[key] = anns
        return result


def load_coco_ground_truth(annotation_path: Path) -> CocoGroundTruth:
    """COCO アノテーション JSON を読み込み, GT データを構築する.

    Args:
        annotation_path: COCO フォーマットのアノテーション JSON パス.

    Returns:
        構築された CocoGroundTruth.

    Examples:
        >>> gt = load_coco_ground_truth(Path("data/val/annotations.json"))
        >>> gt.image_id_by_filename["image_001.jpg"]
        1
        >>> gt.category_id_to_idx
        {1: 0, 2: 1, 3: 2}
    """
    with open(annotation_path, encoding="utf-8") as f:
        coco: dict[str, Any] = json.load(f)

    image_id_by_filename: dict[str, int] = {}
    filenames_by_image_id: dict[int, list[str]] = {}

    for img in coco["images"]:
        file_name = img["file_name"]
        image_id = img["id"]
        image_id_by_filename[file_name] = image_id
        filenames_for_id = [file_name]

        basename = extract_basename(file_name)
        if basename != file_name:
            if basename in image_id_by_filename:
                logger.warning(
                    f"basename '{basename}' が重複しています "
                    f"('{file_name}' と既存エントリ). "
                    f"フルパスでのマッチングのみ使用します."
                )
            else:
                image_id_by_filename[basename] = image_id
                filenames_for_id.append(basename)
        filenames_by_image_id[image_id] = filenames_for_id

    categories = filter_categories(coco.get("categories", []))
    category_id_to_idx = build_category_id_to_idx(categories)

    gt_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in category_id_to_idx:
            continue
        gt_by_image_id.setdefault(ann["image_id"], []).append(ann)

    return CocoGroundTruth(
        image_id_by_filename=image_id_by_filename,
        filenames_by_image_id=filenames_by_image_id,
        category_id_to_idx=category_id_to_idx,
        categories=categories,
        gt_by_image_id=gt_by_image_id,
    )
