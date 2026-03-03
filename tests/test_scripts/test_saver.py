"""InferenceSaver のテスト."""

from pathlib import Path

from pochidetection.scripts.rtdetr.inference.saver import InferenceSaver


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
