"""ConfigLoaderのテスト."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from pochidetection.utils import ConfigLoader


class TestConfigLoader:
    """ConfigLoaderのテスト."""

    def test_load_config(self, tmp_path: Path) -> None:
        """設定ファイルを読み込めることを確認."""
        config_file = tmp_path / "test_config.py"
        config_file.write_text(
            'data_root = "data"\n'
            "num_classes = 2\n"
            'architecture = "RTDetr"\n'
            'train_split = "train"\n'
            'val_split = "val"\n',
            encoding="utf-8",
        )

        config = ConfigLoader.load(str(config_file))
        assert config["num_classes"] == 2
        assert config["architecture"] == "RTDetr"
        assert config["data_root"] == "data"

    def test_load_config_file_not_found(self) -> None:
        """存在しないファイルでエラーが発生することを確認."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load("/nonexistent/path/config.py")

    def test_load_config_missing_required(self, tmp_path: Path) -> None:
        """必須キーがない場合にエラーが発生することを確認."""
        config_file = tmp_path / "bad_config.py"
        config_file.write_text('architecture = "RTDetr"\n', encoding="utf-8")

        with pytest.raises(ValidationError, match="data_root"):
            ConfigLoader.load(str(config_file))

    def test_load_config_invalid_value(self, tmp_path: Path) -> None:
        """許可されていない値でエラーが発生することを確認."""
        config_file = tmp_path / "invalid_config.py"
        config_file.write_text(
            'data_root = "data"\n'
            "num_classes = 2\n"
            'architecture = "InvalidModel"\n',
            encoding="utf-8",
        )

        with pytest.raises(ValidationError, match="architecture"):
            ConfigLoader.load(str(config_file))

    def test_load_config_invalid_type(self, tmp_path: Path) -> None:
        """型が不正な場合にエラーが発生することを確認."""
        config_file = tmp_path / "type_config.py"
        config_file.write_text(
            'data_root = "data"\n'
            'num_classes = "two"\n'  # intであるべき
            'train_split = "train"\n'
            'val_split = "val"\n',
            encoding="utf-8",
        )

        with pytest.raises(ValidationError, match="num_classes"):
            ConfigLoader.load(str(config_file))

    def test_load_config_applies_defaults(self, tmp_path: Path) -> None:
        """デフォルト値が適用されることを確認."""
        config_file = tmp_path / "minimal_config.py"
        config_file.write_text(
            'data_root = "data"\n' "num_classes = 2\n",
            encoding="utf-8",
        )

        config = ConfigLoader.load(str(config_file))

        assert config["architecture"] == "RTDetr"
        assert config["model_name"] == "PekingU/rtdetr_r50vd"
        assert config["pretrained"] is True
        assert config["image_size"] == {"height": 640, "width": 640}
        assert config["batch_size"] == 4
        assert config["epochs"] == 100
        assert config["train_split"] == "train"
        assert config["val_split"] == "val"
        assert config["device"] == "cuda"
        assert config["work_dir"] == "work_dirs"
        assert config["train_score_threshold"] == 0.2
        assert config["infer_score_threshold"] == 0.5
        assert config["nms_iou_threshold"] == 0.5

    def test_load_config_rejects_unknown_key(self, tmp_path: Path) -> None:
        """未知キーがある場合にエラーが発生することを確認."""
        config_file = tmp_path / "unknown_key_config.py"
        config_file.write_text(
            'data_root = "data"\n' "num_classes = 2\n" "unknown_setting = True\n",
            encoding="utf-8",
        )

        with pytest.raises(ValidationError, match="unknown_setting"):
            ConfigLoader.load(str(config_file))

    def test_load_config_rejects_class_names_mismatch(self, tmp_path: Path) -> None:
        """class_names と num_classes が不整合な場合にエラーが発生することを確認."""
        config_file = tmp_path / "class_names_config.py"
        config_file.write_text(
            'data_root = "data"\n' "num_classes = 2\n" 'class_names = ["dog"]\n',
            encoding="utf-8",
        )

        with pytest.raises(
            ValidationError, match="class_names の要素数は num_classes と一致"
        ):
            ConfigLoader.load(str(config_file))

    def test_load_real_config(self) -> None:
        """実際の設定ファイルを読み込めることを確認."""
        config = ConfigLoader.load("configs/rtdetr_coco.py")

        assert "architecture" in config
        assert "num_classes" in config
        assert "data_root" in config
        assert "image_size" in config
