"""Pydantic 設定スキーマ."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)


class ImageSizeConfig(BaseModel):
    """画像サイズ設定."""

    height: PositiveInt = 640
    width: PositiveInt = 640


class DetectionConfig(BaseModel):
    """RT-DETR 設定スキーマ."""

    model_config = ConfigDict(extra="forbid")

    architecture: Literal["RTDetr"] = "RTDetr"
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

    annotation_path: str | None = None

    work_dir: str = Field(default="work_dirs", min_length=1)

    @model_validator(mode="after")
    def validate_class_names(self) -> "DetectionConfig":
        """class_names と num_classes の整合性を検証."""
        if self.class_names is not None and len(self.class_names) != self.num_classes:
            raise ValueError(
                "class_names の要素数は num_classes と一致する必要があります"
            )
        return self
