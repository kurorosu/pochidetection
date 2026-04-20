"""``infer_debug`` ヘルパーの単体テスト."""

from pathlib import Path

from PIL import Image

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.utils.infer_debug import InferDebugConfig, save_infer_debug_image


class TestSaveInferDebugImage:
    """推論 preprocess デバッグ保存の helper テスト."""

    def test_saves_letterboxed_image_as_jpeg(self, tmp_path: Path) -> None:
        """letterbox=True で target_hw サイズの JPEG が保存される."""
        source = Image.new("RGB", (128, 32), color=(255, 0, 0))
        save_path = tmp_path / "infer_debug" / "infer_0000.jpg"

        save_infer_debug_image(
            source_image=source,
            target_hw=(64, 64),
            letterbox=True,
            save_path=save_path,
        )

        assert save_path.exists()
        with Image.open(save_path) as img:
            # letterbox 結果は target_hw サイズ (width, height) = (64, 64)
            assert img.size == (64, 64)
            assert img.format == "JPEG"

    def test_saves_source_image_when_letterbox_disabled(self, tmp_path: Path) -> None:
        """letterbox=False なら source image をそのままのサイズで保存."""
        source = Image.new("RGB", (128, 96), color=(0, 255, 0))
        save_path = tmp_path / "infer_debug" / "infer_0001.jpg"

        save_infer_debug_image(
            source_image=source,
            target_hw=(64, 64),
            letterbox=False,
            save_path=save_path,
        )

        assert save_path.exists()
        with Image.open(save_path) as img:
            assert img.size == (128, 96)

    def test_saves_source_when_target_hw_none(self, tmp_path: Path) -> None:
        """target_hw=None なら source image をそのまま保存."""
        source = Image.new("RGB", (64, 64), color=(0, 0, 255))
        save_path = tmp_path / "infer_debug" / "infer_0002.jpg"

        save_infer_debug_image(
            source_image=source,
            target_hw=None,
            letterbox=True,
            save_path=save_path,
        )

        assert save_path.exists()
        with Image.open(save_path) as img:
            assert img.size == (64, 64)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """保存先の親ディレクトリが無くても自動作成される."""
        source = Image.new("RGB", (64, 64))
        deep_path = tmp_path / "a" / "b" / "c" / "infer_0000.jpg"

        save_infer_debug_image(
            source_image=source,
            target_hw=None,
            letterbox=False,
            save_path=deep_path,
        )

        assert deep_path.exists()

    def test_from_config_returns_config_when_save_count_positive(
        self, tmp_path: Path
    ) -> None:
        """``infer_debug_save_count > 0`` で ``InferDebugConfig`` が構築される."""
        config: DetectionConfigDict = {
            "infer_debug_save_count": 3,
            "image_size": {"height": 640, "width": 480},
            "letterbox": True,
        }
        result = InferDebugConfig.from_config(config, tmp_path)

        assert result is not None
        assert result.save_count == 3
        assert result.output_dir == tmp_path / "infer_debug"
        assert result.target_hw == (640, 480)
        assert result.letterbox is True

    def test_from_config_returns_none_when_disabled(self, tmp_path: Path) -> None:
        """``infer_debug_save_count=0`` で None を返す."""
        config: DetectionConfigDict = {
            "infer_debug_save_count": 0,
            "image_size": {"height": 640, "width": 640},
            "letterbox": True,
        }
        assert InferDebugConfig.from_config(config, tmp_path) is None

    def test_from_config_defaults_letterbox_true_when_missing(
        self, tmp_path: Path
    ) -> None:
        """``letterbox`` キー未指定時は既定 True として扱う."""
        config: DetectionConfigDict = {
            "infer_debug_save_count": 1,
            "image_size": {"height": 320, "width": 320},
        }
        result = InferDebugConfig.from_config(config, tmp_path)

        assert result is not None
        assert result.letterbox is True

    def test_letterbox_padding_is_zero(self, tmp_path: Path) -> None:
        """letterbox padding 領域のピクセル値が 0 (黒) で埋められる."""
        # 横長 128x32 → target 64x64. scale=0.5, new=(16,64), pad_top=pad_bottom=24.
        # 上下 24 行が padding になるはずで値は (0,0,0).
        source = Image.new("RGB", (128, 32), color=(255, 255, 255))
        save_path = tmp_path / "infer_0000.jpg"

        save_infer_debug_image(
            source_image=source,
            target_hw=(64, 64),
            letterbox=True,
            save_path=save_path,
        )

        with Image.open(save_path) as img:
            # padding 領域 (top row) の中央 pixel が黒に近い (JPEG 圧縮の誤差許容).
            top_pixel = img.getpixel((32, 5))
            assert isinstance(top_pixel, tuple)
            r, g, b = top_pixel[:3]
            assert r < 20 and g < 20 and b < 20
