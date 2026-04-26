"""Pydantic 設定スキーマと TypedDict ミラーの定義.

本モジュールは同じ設定項目を 2 系統で表現する. 役割と使い分けは以下に従う.

``DetectionConfig`` (Pydantic discriminated union)
    **System boundary のバリデーター**. 外部から入ってきた設定値 (Python config
    ファイルや dict) を runtime で検証し, 不正値を早期に弾く責務を持つ.
    ``architecture`` を discriminator として ``RTDetrConfig`` /
    ``SSDLiteConfig`` / ``SSD300Config`` のいずれか 1 variant にディスパッチ
    する. 利用箇所は ``ConfigLoader.load()`` の内部 (validation 用) に限定し,
    関数シグネチャやダウンストリームで引き回さない.

``DetectionConfigDict`` (TypedDict)
    **内部で流通する設定値の型**. ``ConfigLoader.load()`` は Pydantic 検証後に
    ``model_dump()`` で dict 化し, ``DetectionConfigDict`` にキャストして返す.
    CLI / API / pipelines / training など, 検証後の設定を扱うすべての関数
    シグネチャはこの TypedDict を採用する. mypy による静的型チェック (キー名の
    typo 検出, 値の型検査) を目的とする.

判断フロー:
    - 外部入力を受け取り validation する層 → ``DetectionConfig``
    - 検証済み設定を消費する層 (関数シグネチャ全般) → ``DetectionConfigDict``
"""

from typing import Annotated, Any, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    TypeAdapter,
    field_validator,
    model_validator,
)

__all__ = [
    "AugmentTransformConfig",
    "AugmentTransformDict",
    "AugmentationConfig",
    "AugmentationDict",
    "BaseDetectionConfig",
    "DetectionConfig",
    "DetectionConfigAdapter",
    "DetectionConfigDict",
    "ImageSizeConfig",
    "ImageSizeDict",
    "RTDetrConfig",
    "SSD300Config",
    "SSDLiteConfig",
    "normalize_architecture",
]


class AugmentTransformDict(TypedDict, total=False):
    """個別の拡張変換の TypedDict."""

    name: str
    p: float


class AugmentationDict(TypedDict, total=False):
    """Data Augmentation 設定の TypedDict."""

    enabled: bool
    transforms: list[AugmentTransformDict]


class ImageSizeDict(TypedDict):
    """画像サイズの TypedDict."""

    height: int
    width: int


class DetectionConfigDict(TypedDict, total=False):
    """検証済み設定値を内部で流通させる TypedDict.

    ``ConfigLoader.load()`` の戻り値型として用い, CLI / API / pipelines /
    training など検証後の設定を受け取るすべての関数シグネチャはこの型を採用する.
    ランタイムのバリデーションは ``DetectionConfig`` (discriminated union) が
    ``ConfigLoader`` 内部で行い, この TypedDict は mypy による静的型チェック
    (キー名の typo 検出, 値の型検査) を目的とする.

    architecture 別の分岐があっても流通形式は単一 dict に揃え, 下流関数の
    シグネチャ展開を抑える. RT-DETR 専用キー (``model_name`` など) は SSD 系
    では Pydantic 検証で弾かれるため, ここに居ても dict 上には現れない.

    ``total=False`` により全キーが型システム上オプショナルとなるが,
    Pydantic バリデーション済みのため実行時は variant 必須キーが必ず存在する.

    See Also:
        ``DetectionConfig``: 対応する Pydantic discriminated union.
            ``ConfigLoader.load()`` 内部で runtime validation に使う.
    """

    architecture: str
    model_name: str
    pretrained: bool
    local_files_only: bool
    image_size: ImageSizeDict

    data_root: str
    train_split: str
    val_split: str
    batch_size: int
    epochs: int
    learning_rate: float
    num_classes: int
    class_names: list[str] | None

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

    augmentation: AugmentationDict | None

    debug_save_count: int
    infer_debug_save_count: int
    letterbox: bool

    camera_fps: int | None
    camera_resolution: list[int] | None

    work_dir: str

    pipeline_mode: Literal["cpu", "gpu"] | None


class AugmentTransformConfig(BaseModel):
    """個別の拡張変換設定.

    Attributes:
        name: torchvision.transforms.v2 のクラス名.
        p: 適用確率 (変換が自前で p を持たない場合に RandomApply でラップ).
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    p: float = Field(default=1.0, ge=0.0, le=1.0)


class AugmentationConfig(BaseModel):
    """Data Augmentation 設定.

    Attributes:
        enabled: True の場合, 学習時にデータ拡張を適用する.
        transforms: 適用する変換のリスト.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    transforms: list[AugmentTransformConfig] = Field(default_factory=list)


class ImageSizeConfig(BaseModel):
    """画像サイズ設定."""

    height: PositiveInt = 640
    width: PositiveInt = 640


_ARCHITECTURE_ALIASES: dict[str, str] = {
    "rtdetr": "RTDetr",
    "ssd300": "SSD300",
    "ssdlite": "SSDLite",
}


def normalize_architecture(value: str) -> str:
    """Normalize architecture 値を正規ラベル (RTDetr / SSD300 / SSDLite) に揃える.

    Args:
        value: ユーザ入力の architecture 文字列 (case-insensitive).

    Returns:
        正規化済みのラベル. 未知ラベルはそのまま返し, Pydantic 側で
        ``Literal`` 検証エラーとして弾かせる.
    """
    return _ARCHITECTURE_ALIASES.get(value.lower(), value)


class BaseDetectionConfig(BaseModel):
    """全アーキテクチャ共通の物体検出設定.

    architecture 固有のフィールドはサブクラスが追加する. 本クラス自体を直接
    バリデーションには使わず, ``DetectionConfig`` (discriminated union) 経由で
    各 variant に dispatch される.
    """

    model_config = ConfigDict(extra="forbid")

    image_size: ImageSizeConfig = Field(default_factory=ImageSizeConfig)

    data_root: str = Field(min_length=1)
    train_split: str = Field(default="train", min_length=1)
    val_split: str = Field(default="val", min_length=1)
    batch_size: PositiveInt = 4
    epochs: PositiveInt = 100
    learning_rate: PositiveFloat = 1e-4
    num_classes: PositiveInt
    class_names: list[str] | None = None

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

    augmentation: AugmentationConfig | None = None

    debug_save_count: int = Field(
        default=10,
        ge=0,
        description=(
            "1 エポック目に保存するデバッグ画像枚数. augmentation の有無に関わらず "
            "学習画像 (augmentation 適用後, preprocess 前) を先頭から N 枚まで "
            "bbox 付きで保存する. 保存先は ``{work_dir}/{run}/train_debug/``. "
            "letterbox / preprocess の silent bug (padding 色 / アスペクト比 / "
            "label 座標のズレ) を目視で早期検知する目的. 0 で無効."
        ),
    )

    infer_debug_save_count: int = Field(
        default=1,
        ge=0,
        description=(
            "推論時に保存する preprocess 後画像の枚数. letterbox 適用後 (モデル入力と"
            "同形状) の画像を先頭から N 枚まで保存する. bbox は含まず, preprocess の "
            "silent bug を目視検知する目的. 保存先は CLI / WebAPI とも "
            "``work_dirs/<train>/best/inference_NNN/infer_debug/``. 0 で無効."
        ),
    )

    letterbox: bool = Field(
        default=True,
        description=(
            "学習 / 推論 preprocess に letterbox (アスペクト比維持 + padding) "
            "リサイズを適用するかどうか. True (既定) で学習 / 推論とも letterbox, "
            "False で従来の単純 resize に戻る. train/infer のミスマッチを避けるため "
            "単一フラグで両方を制御する."
        ),
    )

    camera_fps: PositiveInt | None = None
    camera_resolution: list[PositiveInt] | None = None

    work_dir: str = Field(default="work_dirs", min_length=1)

    pipeline_mode: Literal["cpu", "gpu"] | None = Field(
        default=None,
        description=(
            "preprocess の経路. None の場合は backend 種別から自動解決 "
            "(PyTorch / TensorRT は 'gpu', ONNX は 'cpu'). "
            "CLI --pipeline での明示指定が config 値を上書きする."
        ),
    )

    @field_validator("camera_resolution", mode="before")
    @classmethod
    def validate_camera_resolution(cls, v: list[int] | None) -> list[int] | None:
        """camera_resolution が [width, height] の2要素であることを検証."""
        if v is not None and len(v) != 2:
            raise ValueError(
                "camera_resolution は [width, height] の2要素で指定してください"
            )
        return v

    @field_validator("early_stopping_patience", mode="before")
    @classmethod
    def normalize_early_stopping_patience(cls, v: int | None) -> int | None:
        """patience=0 を None (無効) に正規化."""
        if v is not None and v <= 0:
            return None
        return v

    @model_validator(mode="after")
    def validate_class_names(self) -> "BaseDetectionConfig":
        """class_names と num_classes の整合性を検証."""
        if self.class_names is not None and len(self.class_names) != self.num_classes:
            raise ValueError(
                "class_names の要素数は num_classes と一致する必要があります"
            )
        return self


class RTDetrConfig(BaseDetectionConfig):
    """RT-DETR 用設定.

    ``model_name`` / ``pretrained`` / ``local_files_only`` の HF 系フィールドは
    本 variant のみが受け付ける.
    """

    architecture: Literal["RTDetr"] = "RTDetr"
    model_name: str = Field(default="PekingU/rtdetr_r50vd", min_length=1)
    pretrained: bool = True
    local_files_only: bool = False


class SSDLiteConfig(BaseDetectionConfig):
    """SSDLite (MobileNetV3) 用設定.

    HF 系フィールド (``model_name`` 等) は持たない. 指定された場合は
    ``extra="forbid"`` により ``ValidationError`` を送出する.
    """

    architecture: Literal["SSDLite"]


class SSD300Config(BaseDetectionConfig):
    """SSD300 (VGG16) 用設定.

    HF 系フィールド (``model_name`` 等) は持たない. 指定された場合は
    ``extra="forbid"`` により ``ValidationError`` を送出する.
    """

    architecture: Literal["SSD300"]


DetectionConfig = Annotated[
    RTDetrConfig | SSDLiteConfig | SSD300Config,
    Field(discriminator="architecture"),
]
"""architecture をキーにした discriminated union.

``ConfigLoader.load()`` 内部で ``DetectionConfigAdapter.validate_python`` 経由で
バリデーションする. 直接 ``DetectionConfig.model_validate`` のような呼び出しは
できない (型エイリアスのため). サブクラスを直接バリデーションすることも可能.
"""

DetectionConfigAdapter: TypeAdapter[RTDetrConfig | SSDLiteConfig | SSD300Config] = (
    TypeAdapter(DetectionConfig)
)
"""``DetectionConfig`` の TypeAdapter.

``Annotated[Union, Field(discriminator=...)]`` は型エイリアスのため
``model_validate`` を持たない. ``ConfigLoader`` はこの adapter 経由で
``validate_python`` を呼び出す.
"""
