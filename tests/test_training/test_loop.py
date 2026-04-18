"""scripts/common/training.py のテスト."""

import logging
from pathlib import Path

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.training.loop import (
    _setup_tensorboard,
    build_early_stopping,
)
from pochidetection.utils import EarlyStopping
from pochidetection.visualization.tensorboard import TensorBoardWriter


class TestBuildEarlyStopping:
    """build_early_stopping のテスト."""

    def test_returns_none_when_patience_not_set(self) -> None:
        """patience が未設定の場合 None を返すことを確認."""
        config: DetectionConfigDict = {}
        assert build_early_stopping(config) is None

    def test_returns_none_when_patience_is_none(self) -> None:
        """patience が None の場合 None を返すことを確認."""
        config: DetectionConfigDict = {"early_stopping_patience": None}
        assert build_early_stopping(config) is None

    def test_returns_early_stopping_with_max_mode(self) -> None:
        """mAP メトリクス指定時に mode=max の EarlyStopping を返すことを確認."""
        config: DetectionConfigDict = {
            "early_stopping_patience": 5,
            "early_stopping_metric": "mAP",
            "early_stopping_min_delta": 0.001,
        }
        result = build_early_stopping(config)
        assert isinstance(result, EarlyStopping)
        assert result.patience == 5

    def test_returns_early_stopping_with_min_mode(self) -> None:
        """val_loss メトリクス指定時に mode=min の EarlyStopping を返すことを確認."""
        config: DetectionConfigDict = {
            "early_stopping_patience": 10,
            "early_stopping_metric": "val_loss",
            "early_stopping_min_delta": 0.0,
        }
        result = build_early_stopping(config)
        assert isinstance(result, EarlyStopping)
        assert result.patience == 10


class TestSetupTensorboard:
    """_setup_tensorboard のテスト."""

    def test_returns_none_when_disabled(self, tmp_path: Path) -> None:
        """enable_tensorboard=False の場合 None を返すことを確認."""
        config: DetectionConfigDict = {"enable_tensorboard": False}
        logger = logging.getLogger("test")
        assert _setup_tensorboard(config, tmp_path, logger) is None

    def test_returns_none_when_not_set(self, tmp_path: Path) -> None:
        """enable_tensorboard 未設定の場合 None を返すことを確認."""
        config: DetectionConfigDict = {}
        logger = logging.getLogger("test")
        assert _setup_tensorboard(config, tmp_path, logger) is None

    def test_returns_writer_when_enabled(self, tmp_path: Path) -> None:
        """enable_tensorboard=True の場合 TensorBoardWriter を返すことを確認."""
        config: DetectionConfigDict = {"enable_tensorboard": True}
        logger = logging.getLogger("test")
        writer = _setup_tensorboard(config, tmp_path, logger)
        assert isinstance(writer, TensorBoardWriter)
        writer.close()

    def test_creates_tensorboard_directory_with_workspace_name(
        self, tmp_path: Path
    ) -> None:
        """ワークスペース名をサブディレクトリとして tensorboard ディレクトリが作成されることを確認."""
        workspace = tmp_path / "20260315_001"
        workspace.mkdir()
        config: DetectionConfigDict = {"enable_tensorboard": True}
        logger = logging.getLogger("test")
        writer = _setup_tensorboard(config, workspace, logger)
        assert writer is not None
        assert (workspace / "tensorboard" / "20260315_001").exists()
        writer.close()
