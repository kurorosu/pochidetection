"""検出結果を画像に描画するクラス."""

from PIL import Image, ImageDraw, ImageFont

from pochidetection.scripts.rtdetr.inference.detection import Detection
from pochidetection.visualization import ColorPalette, LabelMapper


class Visualizer:
    """検出結果を画像に描画.

    Attributes:
        _palette: カラーパレット.
        _mapper: ラベルマッパー.
    """

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

    def draw(
        self,
        image: Image.Image,
        detections: list[Detection],
    ) -> Image.Image:
        """検出結果を画像に描画.

        Args:
            image: 入力画像 (PIL Image).
            detections: 検出結果のリスト.

        Returns:
            描画済み画像.
        """
        # 元画像をコピー
        result = image.copy()
        draw = ImageDraw.Draw(result)
        width, height = result.size

        # 画像サイズに応じた線の太さとフォントサイズ
        base_size = max(width, height)
        line_width = max(2, int(base_size / 300))
        font_size = max(12, int(base_size / 40))

        # フォント設定
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

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

            # テキスト描画 (白文字) - 背景の中央に配置
            text_x = bg_x1 + padding - text_left
            text_y = bg_y1 + padding - text_top
            draw.text((text_x, text_y), text, fill="white", font=font)

        return result
