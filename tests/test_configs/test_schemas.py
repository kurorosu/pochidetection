"""DetectionConfig (discriminated union) のテスト."""

import pytest
from pydantic import ValidationError

from pochidetection.configs.schemas import (
    DetectionConfigAdapter,
    RTDetrConfig,
    SSD300Config,
    SSDLiteConfig,
    normalize_architecture,
)

REQUIRED_FIELDS: dict = {
    "architecture": "RTDetr",
    "data_root": "data",
    "num_classes": 4,
}


def _validate(**fields: object) -> RTDetrConfig | SSDLiteConfig | SSD300Config:
    """``DetectionConfigAdapter.validate_python`` のテスト用ショートハンド."""
    return DetectionConfigAdapter.validate_python(fields)


class TestDetectionConfigEarlyStopping:
    """Early Stopping 関連フィールドのテスト."""

    def test_default_early_stopping_disabled(self) -> None:
        """デフォルトでは Early Stopping が無効."""
        config = _validate(**REQUIRED_FIELDS)
        assert config.early_stopping_patience is None
        assert config.early_stopping_metric == "mAP"
        assert config.early_stopping_min_delta == 0.0

    def test_patience_none_disables(self) -> None:
        """patience=None で無効."""
        config = _validate(**REQUIRED_FIELDS, early_stopping_patience=None)
        assert config.early_stopping_patience is None

    def test_patience_zero_normalized_to_none(self) -> None:
        """patience=0 は None に正規化される."""
        config = _validate(**REQUIRED_FIELDS, early_stopping_patience=0)
        assert config.early_stopping_patience is None

    def test_negative_patience_normalized_to_none(self) -> None:
        """負の patience は None に正規化される."""
        config = _validate(**REQUIRED_FIELDS, early_stopping_patience=-1)
        assert config.early_stopping_patience is None

    def test_positive_patience_accepted(self) -> None:
        """正の patience はそのまま受け入れる."""
        config = _validate(**REQUIRED_FIELDS, early_stopping_patience=10)
        assert config.early_stopping_patience == 10

    def test_metric_mAP(self) -> None:
        """metric=mAP を指定できる."""
        config = _validate(
            **REQUIRED_FIELDS,
            early_stopping_patience=5,
            early_stopping_metric="mAP",
        )
        assert config.early_stopping_metric == "mAP"

    def test_metric_val_loss(self) -> None:
        """metric=val_loss を指定できる."""
        config = _validate(
            **REQUIRED_FIELDS,
            early_stopping_patience=5,
            early_stopping_metric="val_loss",
        )
        assert config.early_stopping_metric == "val_loss"

    def test_invalid_metric_raises_error(self) -> None:
        """無効な metric で ValidationError."""
        with pytest.raises(ValidationError):
            _validate(
                **REQUIRED_FIELDS,
                early_stopping_metric="invalid",
            )

    def test_negative_min_delta_raises_error(self) -> None:
        """負の min_delta で ValidationError."""
        with pytest.raises(ValidationError):
            _validate(
                **REQUIRED_FIELDS,
                early_stopping_min_delta=-0.1,
            )

    def test_min_delta_zero_accepted(self) -> None:
        """min_delta=0.0 は受け入れる."""
        config = _validate(**REQUIRED_FIELDS, early_stopping_min_delta=0.0)
        assert config.early_stopping_min_delta == 0.0


class TestArchitectureDiscriminator:
    """architecture を discriminator にした dispatch 動作のテスト."""

    def test_rtdetr_dispatches_to_rtdetr_config(self) -> None:
        """architecture=RTDetr で RTDetrConfig が選ばれる."""
        config = _validate(**REQUIRED_FIELDS)
        assert isinstance(config, RTDetrConfig)
        assert config.architecture == "RTDetr"

    def test_ssdlite_dispatches_to_ssdlite_config(self) -> None:
        """architecture=SSDLite で SSDLiteConfig が選ばれる."""
        config = _validate(
            **{**REQUIRED_FIELDS, "architecture": "SSDLite"},
        )
        assert isinstance(config, SSDLiteConfig)
        assert config.architecture == "SSDLite"

    def test_ssd300_dispatches_to_ssd300_config(self) -> None:
        """architecture=SSD300 で SSD300Config が選ばれる."""
        config = _validate(
            **{**REQUIRED_FIELDS, "architecture": "SSD300"},
        )
        assert isinstance(config, SSD300Config)
        assert config.architecture == "SSD300"

    def test_invalid_architecture_raises_error(self) -> None:
        """無効な architecture で ValidationError."""
        with pytest.raises(ValidationError):
            _validate(
                **{**REQUIRED_FIELDS, "architecture": "InvalidArch"},
            )


class TestArchitectureNormalization:
    """architecture 正規化ヘルパのテスト (loader 層で呼ばれる)."""

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            ("RTDetr", "RTDetr"),
            ("rtdetr", "RTDetr"),
            ("RTDETR", "RTDetr"),
            ("RtDetr", "RTDetr"),
            ("SSDLite", "SSDLite"),
            ("ssdlite", "SSDLite"),
            ("SSDLITE", "SSDLite"),
            ("Ssdlite", "SSDLite"),
            ("SSD300", "SSD300"),
            ("ssd300", "SSD300"),
        ],
    )
    def test_case_insensitive_normalization(
        self, input_value: str, expected: str
    ) -> None:
        """大文字小文字を問わず正規化されることを確認."""
        assert normalize_architecture(input_value) == expected

    def test_unknown_value_returned_as_is(self) -> None:
        """未知ラベルはそのまま返り Pydantic 側で弾かれる."""
        assert normalize_architecture("InvalidArch") == "InvalidArch"


class TestSSDRejectsRTDetrFields:
    """SSD 系 variant が RT-DETR 専用フィールドを拒否することのテスト (#604)."""

    @pytest.mark.parametrize("architecture", ["SSDLite", "SSD300"])
    def test_model_name_raises_validation_error(self, architecture: str) -> None:
        """SSD 系で model_name 指定すると ValidationError (警告ではない)."""
        with pytest.raises(ValidationError) as exc_info:
            _validate(
                **{**REQUIRED_FIELDS, "architecture": architecture},
                model_name="custom_model",
            )
        assert "model_name" in str(exc_info.value)

    @pytest.mark.parametrize("architecture", ["SSDLite", "SSD300"])
    def test_pretrained_raises_validation_error(self, architecture: str) -> None:
        """SSD 系で pretrained 指定すると ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _validate(
                **{**REQUIRED_FIELDS, "architecture": architecture},
                pretrained=False,
            )
        assert "pretrained" in str(exc_info.value)

    @pytest.mark.parametrize("architecture", ["SSDLite", "SSD300"])
    def test_local_files_only_raises_validation_error(self, architecture: str) -> None:
        """SSD 系で local_files_only 指定すると ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _validate(
                **{**REQUIRED_FIELDS, "architecture": architecture},
                local_files_only=True,
            )
        assert "local_files_only" in str(exc_info.value)

    def test_ssd_with_only_common_fields_passes(self) -> None:
        """SSD 系でも共通フィールドだけなら問題なく検証 pass."""
        config = _validate(
            **{**REQUIRED_FIELDS, "architecture": "SSDLite"},
            nms_iou_threshold=0.3,
        )
        assert isinstance(config, SSDLiteConfig)
        assert config.nms_iou_threshold == 0.3


class TestRTDetrAcceptsAllFields:
    """RT-DETR variant が HF 系フィールドを受け付けることのテスト."""

    def test_model_name_accepted(self) -> None:
        """RT-DETR では model_name を受け付ける."""
        config = _validate(**REQUIRED_FIELDS, model_name="custom_model")
        assert isinstance(config, RTDetrConfig)
        assert config.model_name == "custom_model"

    def test_pretrained_false_accepted(self) -> None:
        """RT-DETR では pretrained=False を受け付ける."""
        config = _validate(**REQUIRED_FIELDS, pretrained=False)
        assert isinstance(config, RTDetrConfig)
        assert config.pretrained is False

    def test_local_files_only_accepted(self) -> None:
        """RT-DETR では local_files_only を受け付ける."""
        config = _validate(**REQUIRED_FIELDS, local_files_only=True)
        assert isinstance(config, RTDetrConfig)
        assert config.local_files_only is True


class TestInferImageDir:
    """infer_image_dir フィールドのテスト."""

    def test_default_is_none(self) -> None:
        """デフォルトでは None."""
        config = _validate(**REQUIRED_FIELDS)
        assert config.infer_image_dir is None

    def test_set_infer_image_dir(self) -> None:
        """infer_image_dir を設定できる."""
        config = _validate(**REQUIRED_FIELDS, infer_image_dir="data/val/JPEGImages")
        assert config.infer_image_dir == "data/val/JPEGImages"

    def test_existing_configs_without_field_still_valid(self) -> None:
        """infer_image_dir 未指定の既存 config でもバリデーションが通る."""
        config = _validate(**REQUIRED_FIELDS)
        assert config.infer_image_dir is None
