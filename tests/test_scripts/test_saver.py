"""InferenceSaver のテスト."""

from pathlib import Path

from PIL import Image

from pochidetection.core.detection import Detection
from pochidetection.scripts.common.saver import InferenceSaver
from pochidetection.visualization.label_mapper import LabelMapper


class TestCreateNumberedDir:
    """_create_numbered_dir の連番生成テスト."""

    def test_first_dir_is_001(self, tmp_path: Path) -> None:
        """初回は inference_001 が作成されることを確認."""
        saver = InferenceSaver(tmp_path)
        assert saver.output_dir.name == "inference_001"
        assert saver.output_dir.exists()

    def test_increments_from_existing(self, tmp_path: Path) -> None:
        """既存ディレクトリの次の番号が作成されることを確認."""
        (tmp_path / "inference_001").mkdir()
        (tmp_path / "inference_002").mkdir()
        saver = InferenceSaver(tmp_path)
        assert saver.output_dir.name == "inference_003"

    def test_handles_four_digit_numbers(self, tmp_path: Path) -> None:
        """4桁以上のディレクトリ番号を正しく検出することを確認."""
        (tmp_path / "inference_999").mkdir()
        (tmp_path / "inference_1000").mkdir()
        saver = InferenceSaver(tmp_path)
        assert saver.output_dir.name == "inference_1001"

    def test_ignores_non_matching_dirs(self, tmp_path: Path) -> None:
        """inference_ プレフィックスを持たないディレクトリを無視することを確認."""
        (tmp_path / "other_dir").mkdir()
        (tmp_path / "inference_abc").mkdir()
        saver = InferenceSaver(tmp_path)
        assert saver.output_dir.name == "inference_001"

    def test_base_dir_not_exists(self, tmp_path: Path) -> None:
        """base_dir が存在しない場合に自動作成されることを確認."""
        non_existent = tmp_path / "nested" / "path"
        saver = InferenceSaver(non_existent)
        assert saver.output_dir.name == "inference_001"
        assert non_existent.exists()


class TestSaveCrops:
    """save_crops のテスト."""

    def _create_image(self) -> Image.Image:
        """100x100 のテスト用 RGB 画像を生成."""
        return Image.new("RGB", (100, 100), color=(128, 64, 32))

    def _create_detections(self) -> list[Detection]:
        """テスト用の検出結果を生成."""
        return [
            Detection(box=[10.0, 20.0, 50.0, 60.0], score=0.95, label=0),
            Detection(box=[60.0, 10.0, 90.0, 80.0], score=0.87, label=1),
        ]

    def test_crops_saved_to_crop_dir(self, tmp_path: Path) -> None:
        """クロップ画像が crop/ サブフォルダに保存されることを確認."""
        saver = InferenceSaver(tmp_path)
        image = self._create_image()
        detections = self._create_detections()

        saved = saver.save_crops(image, detections, "test.jpg")

        crop_dir = saver.output_dir / "crop"
        assert crop_dir.exists()
        assert len(saved) == 2
        assert all(p.parent == crop_dir for p in saved)

    def test_crop_filename_format(self, tmp_path: Path) -> None:
        """ファイル名に検出番号とスコアが含まれることを確認."""
        saver = InferenceSaver(tmp_path)
        image = self._create_image()
        detections = [Detection(box=[10, 20, 50, 60], score=0.95, label=0)]

        saved = saver.save_crops(image, detections, "photo.jpg")

        assert saved[0].name == "photo_0_0_0.95.jpg"

    def test_crop_with_label_mapper(self, tmp_path: Path) -> None:
        """LabelMapper 指定時にクラス名がファイル名に含まれることを確認."""
        saver = InferenceSaver(tmp_path)
        image = self._create_image()
        detections = [Detection(box=[10, 20, 50, 60], score=0.92, label=0)]
        mapper = LabelMapper(["person", "cat"])

        saved = saver.save_crops(image, detections, "img.jpg", mapper)

        assert saved[0].name == "img_0_person_0.92.jpg"

    def test_crop_image_size(self, tmp_path: Path) -> None:
        """クロップ画像のサイズが検出ボックスと一致することを確認."""
        saver = InferenceSaver(tmp_path)
        image = self._create_image()
        detections = [Detection(box=[10, 20, 50, 60], score=0.9, label=0)]

        saved = saver.save_crops(image, detections, "test.jpg")

        crop = Image.open(saved[0])
        assert crop.size == (40, 40)  # (50-10, 60-20)

    def test_empty_detections_returns_empty(self, tmp_path: Path) -> None:
        """検出なしの場合は空リストを返し crop/ を作成しないことを確認."""
        saver = InferenceSaver(tmp_path)
        image = self._create_image()

        saved = saver.save_crops(image, [], "test.jpg")

        assert saved == []
        assert not (saver.output_dir / "crop").exists()

    def test_bbox_clipped_to_image_bounds(self, tmp_path: Path) -> None:
        """画像外にはみ出す bbox がクリッピングされることを確認."""
        saver = InferenceSaver(tmp_path)
        image = self._create_image()  # 100x100
        detections = [Detection(box=[-10, -5, 50, 60], score=0.9, label=0)]

        saved = saver.save_crops(image, detections, "test.jpg")

        assert len(saved) == 1
        crop = Image.open(saved[0])
        assert crop.size == (50, 60)  # (min(100,50)-max(0,-10), min(100,60)-max(0,-5))

    def test_bbox_fully_outside_skipped(self, tmp_path: Path) -> None:
        """画像外に完全にはみ出す bbox がスキップされることを確認."""
        saver = InferenceSaver(tmp_path)
        image = self._create_image()  # 100x100
        detections = [Detection(box=[200, 200, 300, 300], score=0.9, label=0)]

        saved = saver.save_crops(image, detections, "test.jpg")

        assert saved == []
