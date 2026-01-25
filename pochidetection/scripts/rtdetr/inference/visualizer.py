"""検出結果を画像に描画するクラス."""

from PIL import Image, ImageDraw, ImageFont

from pochidetection.scripts.rtdetr.inference.detection import Detection


class Visualizer:
    """検出結果を画像に描画.

    Attributes:
        _color: バウンディングボックスの色.
    """

    def __init__(self, color: str = "red") -> None:
        """Visualizerを初期化.

        Args:
            color: バウンディングボックスの色.
        """
        self._color = color

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

            # ボックス描画
            draw.rectangle([x1, y1, x2, y2], outline=self._color, width=line_width)

            # ラベルと信頼度のテキスト
            text = f"{detection.label}: {detection.score:.2f}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # テキスト背景 (ボックス左上)
            padding = 2
            bg_x1 = x1
            bg_y1 = max(0, y1 - text_height - padding * 2)
            bg_x2 = x1 + text_width + padding * 2
            bg_y2 = y1
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=self._color)

            # テキスト描画 (白文字)
            draw.text((x1 + padding, bg_y1 + padding), text, fill="white", font=font)

        return result
