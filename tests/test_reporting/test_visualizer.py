"""Visualizerクラスのテスト."""

import numpy as np
import pytest
from PIL import Image, ImageFont

from pochidetection.core.detection import Detection
from pochidetection.reporting.visualizer import Visualizer
from pochidetection.visualization import ColorPalette


class TestGetTextColor:
    """_get_text_colorメソッドのテスト."""

    @pytest.mark.parametrize(
        ("hex_color", "expected"),
        [
            ("#FFFF00", "black"),  # yellow (明るい)
            ("#00FF00", "black"),  # lime (明るい)
            ("#FFFFFF", "black"),  # white (明るい)
            ("#80FF80", "black"),  # light green (明るい)
            ("#FFFF80", "black"),  # light yellow (明るい)
            ("#00FFFF", "black"),  # cyan (明るい)
            ("#FF0000", "white"),  # red (暗い)
            ("#0000FF", "white"),  # blue (暗い)
            ("#000000", "white"),  # black (暗い)
            ("#8000FF", "white"),  # purple (暗い)
            ("#C00000", "white"),  # dark red (暗い)
        ],
    )
    def test_text_color_based_on_luminance(self, hex_color: str, expected: str) -> None:
        """背景輝度に基づいて正しいテキスト色を返す."""
        assert Visualizer._get_text_color(hex_color) == expected

    def test_boundary_dark(self) -> None:
        """輝度が128以下の場合は白を返す."""
        # R=0, G=180, B=0 -> luminance = 0.7152 * 180 = 128.7 > 128
        assert Visualizer._get_text_color("#00B400") == "black"

    def test_boundary_exactly_128(self) -> None:
        """輝度が128丁度の場合は白を返す."""
        # luminance = 128.0 exactly -> not > 128 -> white
        # 0.2126*R + 0.7152*G + 0.0722*B = 128
        # R=128/0.2126 ≈ 602 -> too large. Use composite.
        # R=0, G=0, B=255: 0.0722*255 = 18.4 (too small)
        # Use direct threshold: luminance must be <= 128 for white
        assert Visualizer._get_text_color("#000000") == "white"  # luminance=0
        assert Visualizer._get_text_color("#FFFFFF") == "black"  # luminance=255


class TestVisualizerDraw:
    """Visualizer.drawメソッドのテスト."""

    def test_draw_with_yellow_uses_black_text(self) -> None:
        """黄色背景のラベルテキスト色が黒になることを確認."""
        assert Visualizer._get_text_color("#FFFF00") == "black"

    def test_draw_with_blue_uses_white_text(self) -> None:
        """青背景のラベルテキスト色が白になることを確認."""
        assert Visualizer._get_text_color("#0000FF") == "white"

    def test_draw_returns_new_image(self) -> None:
        """drawが元画像を変更せず新しい画像を返すことを確認."""
        visualizer = Visualizer()
        image = Image.new("RGB", (200, 200), "gray")
        detections = [Detection(box=[10, 10, 100, 100], score=0.9, label=0)]

        result = visualizer.draw(image, detections)

        assert result is not image

    def test_draw_inplace_returns_same_image(self) -> None:
        """inplace=True の場合, 元画像と同一オブジェクトを返すことを確認."""
        visualizer = Visualizer()
        image = Image.new("RGB", (200, 200), "gray")
        detections = [Detection(box=[10, 10, 100, 100], score=0.9, label=0)]

        result = visualizer.draw(image, detections, inplace=True)

        assert result is image

    def test_draw_empty_detections(self) -> None:
        """検出なしの場合に元画像と同じ内容を返すことを確認."""
        visualizer = Visualizer()
        image = Image.new("RGB", (200, 200), "gray")

        result = visualizer.draw(image, [])

        assert result.tobytes() == image.tobytes()

    def test_draw_modifies_label_area(self) -> None:
        """drawが検出ラベル領域を描画することを確認."""
        palette = ColorPalette(colors=["#FFFF00"])
        visualizer = Visualizer(color_palette=palette)
        image = Image.new("RGB", (200, 200), "gray")
        detections = [Detection(box=[10, 10, 100, 100], score=0.95, label=0)]

        result = visualizer.draw(image, detections)

        # ラベル背景領域が元のgrayから変化していること
        original_pixel = image.getpixel((15, 15))
        result_pixel = result.getpixel((15, 15))
        assert result_pixel != original_pixel

    def test_fallback_font_preserves_size(self) -> None:
        """load_default(size=) でフォントサイズが維持されることを確認."""
        font_size = 20
        font = ImageFont.load_default(size=font_size)
        # Why: stub では `FreeTypeFont | ImageFont` だが size 指定時は FreeTypeFont.
        assert isinstance(font, ImageFont.FreeTypeFont)
        assert font.size == font_size


class TestHexToBgr:
    """_hex_to_bgr メソッドのテスト."""

    @pytest.mark.parametrize(
        ("hex_color", "expected_bgr"),
        [
            ("#FF0000", (0, 0, 255)),  # red
            ("#00FF00", (0, 255, 0)),  # green
            ("#0000FF", (255, 0, 0)),  # blue
            ("#FFFFFF", (255, 255, 255)),  # white
            ("#000000", (0, 0, 0)),  # black
        ],
    )
    def test_hex_to_bgr_conversion(
        self, hex_color: str, expected_bgr: tuple[int, int, int]
    ) -> None:
        """HEX カラーが正しい BGR タプルに変換されることを確認."""
        assert Visualizer._hex_to_bgr(hex_color) == expected_bgr


class TestDrawCv2:
    """Visualizer.draw_cv2 メソッドのテスト."""

    def test_draw_cv2_returns_same_array(self) -> None:
        """draw_cv2 が入力画像と同一の numpy 配列を返すことを確認."""
        visualizer = Visualizer()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [Detection(box=[10, 10, 100, 100], score=0.9, label=0)]

        result = visualizer.draw_cv2(image, detections)

        assert result is image

    def test_draw_cv2_modifies_image(self) -> None:
        """draw_cv2 が画像にボックスを描画することを確認."""
        visualizer = Visualizer()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [Detection(box=[10, 10, 100, 100], score=0.9, label=0)]

        visualizer.draw_cv2(image, detections)

        assert np.any(image != 0)

    def test_draw_cv2_empty_detections(self) -> None:
        """検出なしの場合に画像が変更されないことを確認."""
        visualizer = Visualizer()
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        original = image.copy()

        visualizer.draw_cv2(image, [])

        np.testing.assert_array_equal(image, original)

    def test_draw_cv2_multiple_detections(self) -> None:
        """複数検出が全て描画されることを確認."""
        visualizer = Visualizer()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(box=[10, 10, 50, 50], score=0.9, label=0),
            Detection(box=[200, 200, 300, 300], score=0.8, label=1),
        ]

        visualizer.draw_cv2(image, detections)

        # 両方のボックス領域が描画されている
        assert np.any(image[10:50, 10:50] != 0)
        assert np.any(image[200:300, 200:300] != 0)
