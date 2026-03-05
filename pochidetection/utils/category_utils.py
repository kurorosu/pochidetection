"""COCO カテゴリのフィルタリング・マッピングユーティリティ."""

from typing import Any

BACKGROUND_NAMES: frozenset[str] = frozenset({"_background_", "background"})
"""背景クラスとして除外するカテゴリ名 (小文字)."""


def filter_categories(categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """背景カテゴリを除外し, カテゴリ ID の昇順でソートする.

    Args:
        categories: COCO アノテーションの categories リスト.

    Returns:
        背景除外・ID 昇順ソート済みのカテゴリリスト.
    """
    return sorted(
        [c for c in categories if c["name"].lower() not in BACKGROUND_NAMES],
        key=lambda c: c["id"],
    )


def build_category_id_to_idx(
    categories: list[dict[str, Any]],
) -> dict[int, int]:
    """カテゴリ ID から連続インデックスへのマッピングを生成する.

    Args:
        categories: フィルタリング・ソート済みのカテゴリリスト.

    Returns:
        カテゴリ ID をキー, 連続インデックスを値とする辞書.
    """
    return {cat["id"]: idx for idx, cat in enumerate(categories)}
