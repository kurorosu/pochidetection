"""category_utils のテスト."""

from pochidetection.utils.category_utils import (
    BACKGROUND_NAMES,
    build_category_id_to_idx,
    filter_categories,
)


class TestFilterCategories:
    """filter_categories のテスト."""

    def test_excludes_background(self) -> None:
        """background / _background_ カテゴリが除外される."""
        categories = [
            {"id": 0, "name": "background"},
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "_background_"},
            {"id": 3, "name": "dog"},
        ]
        result = filter_categories(categories)
        names = [c["name"] for c in result]
        assert names == ["cat", "dog"]

    def test_excludes_background_case_insensitive(self) -> None:
        """大文字・小文字を区別せず background を除外する."""
        categories = [
            {"id": 0, "name": "Background"},
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "_BACKGROUND_"},
        ]
        result = filter_categories(categories)
        assert len(result) == 1
        assert result[0]["name"] == "cat"

    def test_sorts_by_id(self) -> None:
        """カテゴリ ID の昇順でソートされる."""
        categories = [
            {"id": 3, "name": "dog"},
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "bird"},
        ]
        result = filter_categories(categories)
        assert [c["id"] for c in result] == [1, 2, 3]

    def test_empty_input(self) -> None:
        """空リストを渡すと空リストが返る."""
        assert filter_categories([]) == []

    def test_all_background(self) -> None:
        """全カテゴリが background の場合, 空リストが返る."""
        categories = [
            {"id": 0, "name": "background"},
            {"id": 1, "name": "_background_"},
        ]
        assert filter_categories(categories) == []


class TestBuildCategoryIdToIdx:
    """build_category_id_to_idx のテスト."""

    def test_builds_mapping(self) -> None:
        """カテゴリ ID から連続インデックスへのマッピングが正しい."""
        categories = [
            {"id": 1, "name": "cat"},
            {"id": 3, "name": "dog"},
            {"id": 5, "name": "bird"},
        ]
        result = build_category_id_to_idx(categories)
        assert result == {1: 0, 3: 1, 5: 2}

    def test_empty_input(self) -> None:
        """空リストを渡すと空辞書が返る."""
        assert build_category_id_to_idx([]) == {}


class TestBackgroundNames:
    """BACKGROUND_NAMES 定数のテスト."""

    def test_contains_expected_names(self) -> None:
        """background と _background_ が含まれる."""
        assert "background" in BACKGROUND_NAMES
        assert "_background_" in BACKGROUND_NAMES

    def test_is_frozenset(self) -> None:
        """frozenset 型である."""
        assert isinstance(BACKGROUND_NAMES, frozenset)
