"""scripts/common/training.py のテスト."""

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.scripts.common.training import build_early_stopping
from pochidetection.utils import EarlyStopping


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
