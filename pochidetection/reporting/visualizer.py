"""検出結果を画像に描画するクラス."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pochidetection.core.detection import Detection
from pochidetection.visualization import ColorPalette, LabelMapper


class Visualizer:
    """検出結果を画像に描画.

    Attributes:
        _palette: カラーパレット.
        _mapper: ラベルマッパー.
    """

    _LINE_WIDTH_DIVISOR = 300
    _FONT_SIZE_DIVISOR = 40
    _LUMINANCE_THRESHOLD = 128

    def __init__(
        self,
        color_palette: ColorPalette | None = None,
        label_mapper: LabelMapper | None = None,
    ) -> None:
        """Visualizerを初期化.

        Args:
            color_palette: カラーパレット. Noneの場合はデフォルトを使用.
            label_mapper: ラベルマッパー. Noneの場合は整数ラベルを表示.
        """
        self._palette = color_palette if color_palette is not None else ColorPalette()
        self._mapper = label_mapper if label_mapper is not None else LabelMapper()

    @staticmethod
    def _get_text_color(background_hex: str) -> str:
        """背景色の輝度に基づいてテキスト色を決定.

        W3C 相対輝度の公式で背景の明るさを計算し,
        明るい背景には黒, 暗い背景には白を返す.

        Args:
            background_hex: HEX形式の背景色 (例: "#FFFF00").

        Returns:
            テキスト色 ("black" or "white").
        """
        hex_color = background_hex.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "black" if luminance > Visualizer._LUMINANCE_THRESHOLD else "white"

    def draw(
        self,
        image: Image.Image,
        detections: list[Detection],
        *,
        inplace: bool = False,
    ) -> Image.Image:
        """検出結果を画像に描画.

        Args:
            image: 入力画像 (PIL Image).
            detections: 検出結果のリスト.
            inplace: True の場合, 元画像に直接描画しコピーを省略する.

        Returns:
            描画済み画像.
        """
        result = image if inplace else image.copy()
        draw = ImageDraw.Draw(result)
        width, height = result.size

        # 画像サイズに応じた線の太さとフォントサイズ
        base_size = max(width, height)
        line_width = max(2, int(base_size / self._LINE_WIDTH_DIVISOR))
        font_size = max(12, int(base_size / self._FONT_SIZE_DIVISOR))

        # フォント設定
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default(size=font_size)

        for detection in detections:
            x1, y1, x2, y2 = detection.box

            # クラスに応じた色とラベル名を取得
            color = self._palette.get_color(detection.label)
            label_name = self._mapper.get_label(detection.label)

            # ボックス描画
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # ラベルと信頼度のテキスト
            text = f"{label_name}: {detection.score:.2f}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_left, text_top, text_right, text_bottom = text_bbox
            text_width = text_right - text_left
            text_height = text_bottom - text_top

            # テキスト背景 (ボックス内側左上)
            padding = 4
            bg_x1 = x1
            bg_y1 = y1
            bg_x2 = x1 + text_width + padding * 2
            bg_y2 = y1 + text_height + padding * 2
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color)

            # テキスト描画 - 背景の中央に配置
            text_color = self._get_text_color(color)
            text_x = bg_x1 + padding - text_left
            text_y = bg_y1 + padding - text_top
            draw.text((text_x, text_y), text, fill=text_color, font=font)

        return result

    @staticmethod
    def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
        """HEX カラーを BGR タプルに変換.

        Args:
            hex_color: HEX 形式の色 (例: "#FF0000").

        Returns:
            BGR タプル (例: (0, 0, 255)).
        """
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

    def draw_cv2(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """検出結果を OpenCV で直接描画.

        PIL 変換を介さず numpy BGR 画像に直接描画するため,
        リアルタイム推論でのオーバーヘッドを削減する.

        Args:
            image: 入力画像 (numpy BGR).
            detections: 検出結果のリスト.

        Returns:
            描画済み画像 (入力画像を直接変更して返す).
        """
        h, w = image.shape[:2]
        base_size = max(w, h)
        line_width = max(2, int(base_size / self._LINE_WIDTH_DIVISOR))
        font_scale = max(0.4, base_size / self._FONT_SIZE_DIVISOR / 30)
        font_thickness = max(1, line_width // 2)

        for detection in detections:
            x1, y1, x2, y2 = (int(v) for v in detection.box)

            color_hex = self._palette.get_color(detection.label)
            color_bgr = self._hex_to_bgr(color_hex)
            label_name = self._mapper.get_label(detection.label)

            # ボックス描画
            cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, line_width)

            # ラベルテキスト
            text = f"{label_name}: {detection.score:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # テキスト背景
            padding = 4
            bg_y1 = max(0, y1)
            bg_y2 = bg_y1 + th + baseline + padding * 2
            cv2.rectangle(
                image, (x1, bg_y1), (x1 + tw + padding * 2, bg_y2), color_bgr, -1
            )

            # テキスト色 (W3C 輝度)
            text_color_name = self._get_text_color(color_hex)
            text_bgr = (0, 0, 0) if text_color_name == "black" else (255, 255, 255)

            cv2.putText(
                image,
                text,
                (x1 + padding, bg_y1 + padding + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_bgr,
                font_thickness,
            )

        return image
