"""DetectionConfig のテスト."""

import pytest

from pochidetection.configs.schemas import DetectionConfig

REQUIRED_FIELDS: dict = {
    "data_root": "data",
    "num_classes": 4,
}


class TestDetectionConfigEarlyStopping:
    """Early Stopping 関連フィールドのテスト."""

    def test_default_early_stopping_disabled(self) -> None:
        """デフォルトでは Early Stopping が無効."""
        config = DetectionConfig(**REQUIRED_FIELDS)
        assert config.early_stopping_patience is None
        assert config.early_stopping_metric == "mAP"
        assert config.early_stopping_min_delta == 0.0

    def test_patience_none_disables(self) -> None:
        """patience=None で無効."""
        config = DetectionConfig(**REQUIRED_FIELDS, early_stopping_patience=None)
        assert config.early_stopping_patience is None

    def test_patience_zero_normalized_to_none(self) -> None:
        """patience=0 は None に正規化される."""
        config = DetectionConfig(**REQUIRED_FIELDS, early_stopping_patience=0)
        assert config.early_stopping_patience is None

    def test_negative_patience_normalized_to_none(self) -> None:
        """負の patience は None に正規化される."""
        config = DetectionConfig(**REQUIRED_FIELDS, early_stopping_patience=-1)
        assert config.early_stopping_patience is None

    def test_positive_patience_accepted(self) -> None:
        """正の patience はそのまま受け入れる."""
        config = DetectionConfig(**REQUIRED_FIELDS, early_stopping_patience=10)
        assert config.early_stopping_patience == 10

    def test_metric_mAP(self) -> None:
        """metric=mAP を指定できる."""
        config = DetectionConfig(
            **REQUIRED_FIELDS,
            early_stopping_patience=5,
            early_stopping_metric="mAP",
        )
        assert config.early_stopping_metric == "mAP"

    def test_metric_val_loss(self) -> None:
        """metric=val_loss を指定できる."""
        config = DetectionConfig(
            **REQUIRED_FIELDS,
            early_stopping_patience=5,
            early_stopping_metric="val_loss",
        )
        assert config.early_stopping_metric == "val_loss"

    def test_invalid_metric_raises_error(self) -> None:
        """無効な metric で ValidationError."""
        with pytest.raises(Exception):
            DetectionConfig(
                **REQUIRED_FIELDS,
                early_stopping_metric="invalid",
            )

    def test_negative_min_delta_raises_error(self) -> None:
        """負の min_delta で ValidationError."""
        with pytest.raises(Exception):
            DetectionConfig(
                **REQUIRED_FIELDS,
                early_stopping_min_delta=-0.1,
            )

    def test_min_delta_zero_accepted(self) -> None:
        """min_delta=0.0 は受け入れる."""
        config = DetectionConfig(**REQUIRED_FIELDS, early_stopping_min_delta=0.0)
        assert config.early_stopping_min_delta == 0.0
