"""可視化モジュールのテスト."""

import pytest

from pochidetection.visualization import ColorPalette, LabelMapper


class TestColorPalette:
    """ColorPaletteクラスのテスト."""

    def test_default_palette_has_colors(self) -> None:
        """デフォルトパレットが色を持つことを確認."""
        palette = ColorPalette()
        assert len(palette.colors) > 0

    def test_get_color_returns_hex(self) -> None:
        """get_colorがHEX形式を返すことを確認."""
        palette = ColorPalette()
        color = palette.get_color(0)
        assert color.startswith("#")
        assert len(color) == 7

    def test_get_color_cycles_for_large_class_id(self) -> None:
        """大きなクラスIDでも色を返すことを確認 (循環)."""
        palette = ColorPalette()
        num_colors = len(palette.colors)
        # 範囲外のIDでも色を返す
        color = palette.get_color(num_colors + 5)
        assert color.startswith("#")
        # 循環することを確認
        assert palette.get_color(0) == palette.get_color(num_colors)

    def test_custom_palette(self) -> None:
        """カスタムパレットを使用できることを確認."""
        custom_colors = ["#000000", "#FFFFFF"]
        palette = ColorPalette(colors=custom_colors)
        assert palette.colors == custom_colors
        assert palette.get_color(0) == "#000000"
        assert palette.get_color(1) == "#FFFFFF"

    def test_different_class_ids_get_different_colors(self) -> None:
        """異なるクラスIDで異なる色を取得することを確認."""
        palette = ColorPalette()
        colors = [palette.get_color(i) for i in range(5)]
        # 最初の5色は全て異なる
        assert len(set(colors)) == 5


class TestLabelMapper:
    """LabelMapperクラスのテスト."""

    def test_without_class_names_returns_string_id(self) -> None:
        """class_names未設定時は整数を文字列で返すことを確認."""
        mapper = LabelMapper()
        assert mapper.get_label(0) == "0"
        assert mapper.get_label(5) == "5"

    def test_with_class_names_returns_name(self) -> None:
        """class_names設定時はクラス名を返すことを確認."""
        mapper = LabelMapper(class_names=["dog", "cat", "bird"])
        assert mapper.get_label(0) == "dog"
        assert mapper.get_label(1) == "cat"
        assert mapper.get_label(2) == "bird"

    def test_out_of_range_returns_string_id(self) -> None:
        """範囲外のIDは整数を文字列で返すことを確認."""
        mapper = LabelMapper(class_names=["dog", "cat"])
        assert mapper.get_label(2) == "2"
        assert mapper.get_label(10) == "10"

    def test_class_names_property(self) -> None:
        """class_namesプロパティの確認."""
        mapper = LabelMapper(class_names=["dog", "cat"])
        assert mapper.class_names == ["dog", "cat"]

        mapper_empty = LabelMapper()
        assert mapper_empty.class_names is None

    def test_num_classes_property(self) -> None:
        """num_classesプロパティの確認."""
        mapper = LabelMapper(class_names=["dog", "cat", "bird"])
        assert mapper.num_classes == 3

        mapper_empty = LabelMapper()
        assert mapper_empty.num_classes is None
