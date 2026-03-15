"""Pydantic 設定スキーマ."""

import warnings
from typing import Any, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)


class ImageSizeDict(TypedDict):
    """画像サイズの TypedDict."""

    height: int
    width: int


class DetectionConfigDict(TypedDict, total=False):
    """DetectionConfig.model_dump() の出力に対応する TypedDict.

    ConfigLoader.load() の戻り値型として使用する.
    ランタイムのバリデーションは DetectionConfig (Pydantic) が担い,
    この TypedDict は mypy による静的型チェック (キー名の typo 検出,
    値の型検査) を目的とする.

    total=False により全キーがオプショナルとなるが,
    Pydantic バリデーション済みのため実行時は全キーが存在する.
    """

    architecture: str
    model_name: str
    pretrained: bool
    image_size: ImageSizeDict

    data_root: str
    train_split: str
    val_split: str
    batch_size: int
    epochs: int
    learning_rate: float
    num_classes: int
    class_names: list[str] | None

    loss: str
    metrics: str
    dataset: str
    device: str
    cudnn_benchmark: bool
    use_fp16: bool
    enable_tensorboard: bool

    train_score_threshold: float
    infer_score_threshold: float
    nms_iou_threshold: float

    lr_scheduler: str | None
    lr_scheduler_params: dict[str, Any] | None

    early_stopping_patience: int | None
    early_stopping_metric: str
    early_stopping_min_delta: float

    annotation_path: str | None
    infer_image_dir: str | None

    work_dir: str


class ImageSizeConfig(BaseModel):
    """画像サイズ設定."""

    height: PositiveInt = 640
    width: PositiveInt = 640


class DetectionConfig(BaseModel):
    """物体検出設定スキーマ."""

    model_config = ConfigDict(extra="forbid")

    architecture: Literal["RTDetr", "SSDLite"] = "RTDetr"
    model_name: str = Field(default="PekingU/rtdetr_r50vd", min_length=1)
    pretrained: bool = True
    image_size: ImageSizeConfig = Field(default_factory=ImageSizeConfig)

    data_root: str = Field(min_length=1)
    train_split: str = Field(default="train", min_length=1)
    val_split: str = Field(default="val", min_length=1)
    batch_size: PositiveInt = 4
    epochs: PositiveInt = 100
    learning_rate: PositiveFloat = 1e-4
    num_classes: PositiveInt
    class_names: list[str] | None = None

    loss: Literal["DetectionLoss"] = "DetectionLoss"
    metrics: str = Field(default="DetectionMetrics", min_length=1)
    dataset: str = Field(default="CocoDetectionDataset", min_length=1)
    device: Literal["cuda", "cpu"] = "cuda"
    cudnn_benchmark: bool = False
    use_fp16: bool = False
    enable_tensorboard: bool = False

    train_score_threshold: float = Field(default=0.5, ge=0, le=1)
    infer_score_threshold: float = Field(default=0.5, ge=0, le=1)
    nms_iou_threshold: float = Field(default=0.5, ge=0, le=1)

    lr_scheduler: str | None = None
    lr_scheduler_params: dict[str, Any] | None = None

    early_stopping_patience: int | None = None
    early_stopping_metric: Literal["mAP", "val_loss"] = "mAP"
    early_stopping_min_delta: float = Field(default=0.0, ge=0)

    annotation_path: str | None = None
    infer_image_dir: str | None = None

    work_dir: str = Field(default="work_dirs", min_length=1)

    @field_validator("architecture", mode="before")
    @classmethod
    def normalize_architecture(cls, v: str) -> str:
        """Architecture を case-insensitive に正規化."""
        mapping = {
            "rtdetr": "RTDetr",
            "ssdlite": "SSDLite",
        }
        normalized = mapping.get(v.lower())
        if normalized is None:
            return v
        return normalized

    @field_validator("early_stopping_patience", mode="before")
    @classmethod
    def normalize_early_stopping_patience(cls, v: int | None) -> int | None:
        """patience=0 を None (無効) に正規化."""
        if v is not None and v <= 0:
            return None
        return v

    @model_validator(mode="after")
    def validate_class_names(self) -> "DetectionConfig":
        """class_names と num_classes の整合性を検証."""
        if self.class_names is not None and len(self.class_names) != self.num_classes:
            raise ValueError(
                "class_names の要素数は num_classes と一致する必要があります"
            )
        return self

    @model_validator(mode="after")
    def warn_ssdlite_ignored_fields(self) -> "DetectionConfig":
        """Warn about config fields ignored by SSDLite."""
        if self.architecture != "SSDLite":
            return self

        ignored: list[str] = []
        if self.model_name != "PekingU/rtdetr_r50vd":
            ignored.append("model_name")

        if not self.pretrained:
            ignored.append("pretrained")

        for name in ignored:
            warnings.warn(
                f"SSDLite では '{name}' は無視されます.",
                UserWarning,
                stacklevel=2,
            )

        return self
