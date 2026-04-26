"""ConfigLoaderのテスト."""

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from pochidetection.configs.schemas import DetectionConfigDict
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
            'architecture = "RTDetr"\n'
            'data_root = "data"\n'
            'num_classes = "two"\n'  # intであるべき
            'train_split = "train"\n'
            'val_split = "val"\n',
            encoding="utf-8",
        )

        with pytest.raises(ValidationError, match="num_classes"):
            ConfigLoader.load(str(config_file))

    def test_load_config_applies_defaults(self, tmp_path: Path) -> None:
        """architecture 以外のデフォルト値が適用されることを確認."""
        config_file = tmp_path / "minimal_config.py"
        config_file.write_text(
            'architecture = "RTDetr"\n' 'data_root = "data"\n' "num_classes = 2\n",
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
        assert config["train_score_threshold"] == 0.5
        assert config["infer_score_threshold"] == 0.5
        assert config["nms_iou_threshold"] == 0.5
        assert config["lr_scheduler"] is None
        assert config["lr_scheduler_params"] is None

    def test_load_config_missing_architecture(self, tmp_path: Path) -> None:
        """architecture 未指定で discriminator missing エラーが発生することを確認."""
        config_file = tmp_path / "no_arch_config.py"
        config_file.write_text(
            'data_root = "data"\n' "num_classes = 2\n",
            encoding="utf-8",
        )

        with pytest.raises(ValidationError, match="architecture"):
            ConfigLoader.load(str(config_file))

    def test_load_config_rejects_unknown_key(self, tmp_path: Path) -> None:
        """未知キーがある場合にエラーが発生することを確認."""
        config_file = tmp_path / "unknown_key_config.py"
        config_file.write_text(
            'architecture = "RTDetr"\n'
            'data_root = "data"\n'
            "num_classes = 2\n"
            "unknown_setting = True\n",
            encoding="utf-8",
        )

        with pytest.raises(ValidationError, match="unknown_setting"):
            ConfigLoader.load(str(config_file))

    def test_load_config_rejects_class_names_mismatch(self, tmp_path: Path) -> None:
        """class_names と num_classes が不整合な場合にエラーが発生することを確認."""
        config_file = tmp_path / "class_names_config.py"
        config_file.write_text(
            'architecture = "RTDetr"\n'
            'data_root = "data"\n'
            "num_classes = 2\n"
            'class_names = ["dog"]\n',
            encoding="utf-8",
        )

        with pytest.raises(
            ValidationError, match="class_names の要素数は num_classes と一致"
        ):
            ConfigLoader.load(str(config_file))

    def test_score_threshold_accepts_zero(self, tmp_path: Path) -> None:
        """score_threshold に 0.0 を設定できることを確認."""
        config_file = tmp_path / "zero_threshold_config.py"
        config_file.write_text(
            'architecture = "RTDetr"\n'
            'data_root = "data"\n'
            "num_classes = 2\n"
            "train_score_threshold = 0.0\n"
            "infer_score_threshold = 0.0\n",
            encoding="utf-8",
        )

        config = ConfigLoader.load(str(config_file))
        assert config["train_score_threshold"] == 0.0
        assert config["infer_score_threshold"] == 0.0

    def test_load_config_with_lr_scheduler(self, tmp_path: Path) -> None:
        """lr_scheduler を設定できることを確認."""
        config_file = tmp_path / "scheduler_config.py"
        config_file.write_text(
            'architecture = "RTDetr"\n'
            'data_root = "data"\n'
            "num_classes = 2\n"
            'lr_scheduler = "CosineAnnealingLR"\n'
            'lr_scheduler_params = {"eta_min": 1e-6}\n',
            encoding="utf-8",
        )

        config = ConfigLoader.load(str(config_file))
        assert config["lr_scheduler"] == "CosineAnnealingLR"
        assert config["lr_scheduler_params"] == {"eta_min": 1e-6}

    def test_load_real_config(self) -> None:
        """実際の設定ファイルを読み込めることを確認."""
        config = ConfigLoader.load("configs/rtdetr_coco.py")

        assert "architecture" in config
        assert "num_classes" in config
        assert "data_root" in config
        assert "image_size" in config

    def test_load_merges_base_config(self, tmp_path: Path) -> None:
        """_base.py が存在する場合にマージされることを確認."""
        base_file = tmp_path / "_base.py"
        base_file.write_text(
            'architecture = "RTDetr"\n'
            'data_root = "data"\n'
            "num_classes = 2\n"
            'train_split = "train"\n'
            'val_split = "val"\n',
            encoding="utf-8",
        )
        config_file = tmp_path / "model_config.py"
        config_file.write_text(
            "batch_size = 16\n",
            encoding="utf-8",
        )

        config = ConfigLoader.load(str(config_file))

        assert config["data_root"] == "data"
        assert config["num_classes"] == 2
        assert config["batch_size"] == 16

    def test_load_individual_overrides_base(self, tmp_path: Path) -> None:
        """個別設定がベース設定を上書きすることを確認."""
        base_file = tmp_path / "_base.py"
        base_file.write_text(
            'architecture = "RTDetr"\n'
            'data_root = "data"\n'
            "num_classes = 2\n"
            "batch_size = 4\n"
            'train_split = "train"\n'
            'val_split = "val"\n',
            encoding="utf-8",
        )
        config_file = tmp_path / "model_config.py"
        config_file.write_text(
            "batch_size = 32\n",
            encoding="utf-8",
        )

        config = ConfigLoader.load(str(config_file))

        assert config["batch_size"] == 32

    def test_write_config_creates_python_file(self, tmp_path: Path) -> None:
        """write_configがPythonファイルを作成することを確認."""
        config = cast(DetectionConfigDict, {"batch_size": 32, "learning_rate": 0.001})
        output_path = tmp_path / "saved_config.py"

        ConfigLoader.write_config(config, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "batch_size = 32" in content
        assert "learning_rate = 0.001" in content

    def test_write_config_roundtrip(self, tmp_path: Path) -> None:
        """write_configで保存した設定をloadで読み込めることを確認."""
        original = ConfigLoader.load("configs/rtdetr_coco.py")
        saved_path = tmp_path / "roundtrip_config.py"

        ConfigLoader.write_config(original, saved_path)
        reloaded = ConfigLoader.load(str(saved_path))

        assert original == reloaded
