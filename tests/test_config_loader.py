"""ConfigLoaderのテスト."""

import pytest

from pochidetection.utils import ConfigLoader


class TestConfigLoader:
    """ConfigLoaderのテスト."""

    def test_load_config(self, tmp_path: pytest.TempPathFactory) -> None:
        """設定ファイルを読み込めることを確認."""
        config_file = tmp_path / "test_config.py"  # type: ignore
        config_file.write_text(
            'data_root = "data"\n' "num_classes = 2\n" 'architecture = "RTDetr"\n'
        )

        config = ConfigLoader.load(str(config_file))
        assert config["num_classes"] == 2
        assert config["architecture"] == "RTDetr"
        assert config["data_root"] == "data"

    def test_load_config_file_not_found(self) -> None:
        """存在しないファイルでエラーが発生することを確認."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load("/nonexistent/path/config.py")

    def test_load_config_missing_required(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """必須キーがない場合にエラーが発生することを確認."""
        config_file = tmp_path / "bad_config.py"  # type: ignore
        config_file.write_text('architecture = "RTDetr"\n')

        with pytest.raises(KeyError, match="必須キー.*存在しません"):
            ConfigLoader.load(str(config_file))

    def test_load_config_invalid_value(self, tmp_path: pytest.TempPathFactory) -> None:
        """許可されていない値でエラーが発生することを確認."""
        config_file = tmp_path / "invalid_config.py"  # type: ignore
        config_file.write_text(
            'data_root = "data"\n' "num_classes = 2\n" 'architecture = "InvalidModel"\n'
        )

        with pytest.raises(ValueError, match="設定値が不正です"):
            ConfigLoader.load(str(config_file))

    def test_load_config_invalid_type(self, tmp_path: pytest.TempPathFactory) -> None:
        """型が不正な場合にエラーが発生することを確認."""
        config_file = tmp_path / "type_config.py"  # type: ignore
        config_file.write_text(
            'data_root = "data"\n'
            'num_classes = "two"\n'  # intであるべき
        )

        with pytest.raises(TypeError, match="設定値の型が不正です"):
            ConfigLoader.load(str(config_file))

    def test_load_config_applies_defaults(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """デフォルト値が適用されることを確認."""
        config_file = tmp_path / "minimal_config.py"  # type: ignore
        config_file.write_text('data_root = "data"\n' "num_classes = 2\n")

        config = ConfigLoader.load(str(config_file))

        # デフォルト値が適用されているか
        assert config["architecture"] == "RTDetr"
        assert config["model_name"] == "PekingU/rtdetr_r50vd"
        assert config["pretrained"] is True
        assert config["image_size"] == 640
        assert config["batch_size"] == 4
        assert config["epochs"] == 100
        assert config["device"] == "cuda"

    def test_load_real_config(self) -> None:
        """実際の設定ファイルを読み込めることを確認."""
        config = ConfigLoader.load("configs/rtdetr_coco.py")
        # 必須キーが存在することを確認
        assert "architecture" in config
        assert "num_classes" in config
        assert "data_root" in config
        assert "image_size" in config
