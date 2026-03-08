"""DetectionConfig のテスト."""

import warnings

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


class TestArchitectureNormalization:
    """architecture フィールドの case-insensitive 正規化テスト."""

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
        ],
    )
    def test_case_insensitive_normalization(
        self, input_value: str, expected: str
    ) -> None:
        """大文字小文字を問わず正規化されることを確認."""
        config = DetectionConfig(**REQUIRED_FIELDS, architecture=input_value)
        assert config.architecture == expected

    def test_invalid_architecture_raises_error(self) -> None:
        """無効な architecture で ValidationError."""
        with pytest.raises(Exception):
            DetectionConfig(**REQUIRED_FIELDS, architecture="InvalidArch")


class TestSSDLiteIgnoredFieldWarnings:
    """SSDLite で無視される設定項目の警告テスト."""

    SSDLITE_FIELDS: dict = {
        **REQUIRED_FIELDS,
        "architecture": "SSDLite",
    }

    def test_no_warning_with_defaults(self) -> None:
        """デフォルト値ならば警告なし."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DetectionConfig(**self.SSDLITE_FIELDS)
            ssdlite_warnings = [x for x in w if "SSDLite" in str(x.message)]
            assert len(ssdlite_warnings) == 0

    def test_model_name_warns(self) -> None:
        """model_name を変更すると警告."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DetectionConfig(**self.SSDLITE_FIELDS, model_name="custom_model")
            msgs = [str(x.message) for x in w if "SSDLite" in str(x.message)]
            assert any("model_name" in m for m in msgs)

    def test_nms_iou_threshold_warns(self) -> None:
        """nms_iou_threshold を変更すると警告."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DetectionConfig(**self.SSDLITE_FIELDS, nms_iou_threshold=0.3)
            msgs = [str(x.message) for x in w if "SSDLite" in str(x.message)]
            assert any("nms_iou_threshold" in m for m in msgs)

    def test_pretrained_false_warns(self) -> None:
        """pretrained=False で警告."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DetectionConfig(**self.SSDLITE_FIELDS, pretrained=False)
            msgs = [str(x.message) for x in w if "SSDLite" in str(x.message)]
            assert any("pretrained" in m for m in msgs)

    def test_pretrained_true_no_warning(self) -> None:
        """pretrained=True (デフォルト) なら警告なし."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DetectionConfig(**self.SSDLITE_FIELDS, pretrained=True)
            msgs = [str(x.message) for x in w if "pretrained" in str(x.message)]
            assert len(msgs) == 0

    def test_rtdetr_no_warning(self) -> None:
        """RTDetr アーキテクチャでは警告なし."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DetectionConfig(
                **REQUIRED_FIELDS,
                architecture="RTDetr",
                model_name="custom_model",
                nms_iou_threshold=0.3,
            )
            ssdlite_warnings = [x for x in w if "SSDLite" in str(x.message)]
            assert len(ssdlite_warnings) == 0

    def test_multiple_ignored_fields_warn(self) -> None:
        """複数の無視される項目を同時に設定すると各フィールドで警告."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DetectionConfig(
                **self.SSDLITE_FIELDS,
                model_name="custom_model",
                nms_iou_threshold=0.3,
                pretrained=False,
            )
            msgs = [str(x.message) for x in w if "SSDLite" in str(x.message)]
            assert len(msgs) == 3


class TestInferImageDir:
    """infer_image_dir フィールドのテスト."""

    def test_default_is_none(self) -> None:
        """デフォルトでは None."""
        config = DetectionConfig(**REQUIRED_FIELDS)
        assert config.infer_image_dir is None

    def test_set_infer_image_dir(self) -> None:
        """infer_image_dir を設定できる."""
        config = DetectionConfig(
            **REQUIRED_FIELDS, infer_image_dir="data/val/JPEGImages"
        )
        assert config.infer_image_dir == "data/val/JPEGImages"

    def test_existing_configs_without_field_still_valid(self) -> None:
        """infer_image_dir 未指定の既存 config でもバリデーションが通る."""
        config = DetectionConfig(**REQUIRED_FIELDS)
        assert config.infer_image_dir is None
