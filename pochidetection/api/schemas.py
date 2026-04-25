"""API リクエスト/レスポンスのスキーマ定義."""

from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from pochidetection.api.constants import _ALLOWED_DTYPES, MAX_PIXELS


class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス."""

    status: str = Field(description="healthy / unhealthy")
    model_loaded: bool = Field(description="モデルがロード済みかどうか")
    architecture: str | None = Field(default=None, description="モデルアーキテクチャ")


class VersionResponse(BaseModel):
    """バージョン情報レスポンス."""

    pochidetection_version: str
    api_version: str
    backend_versions: dict[str, str] = Field(
        description="検出されたバックエンドのバージョン (torch, onnxruntime, tensorrt 等)"
    )


class ModelInfoResponse(BaseModel):
    """モデル情報レスポンス."""

    architecture: str
    num_classes: int
    class_names: list[str]
    input_size: tuple[int, int] = Field(description="(height, width)")
    model_path: str
    backend: str = Field(description="現在ロードされているバックエンド名")


class BackendsResponse(BaseModel):
    """利用可能バックエンド一覧レスポンス."""

    available: list[str] = Field(description="この環境で利用可能なバックエンド一覧")
    current: str = Field(description="現在ロード中のバックエンド. 未ロード時は 'none'")


class DetectRequest(BaseModel):
    """検出リクエスト. numpy 配列を base64 エンコードして送信する."""

    image_data: str = Field(description="base64 エンコードされた画像データ")
    format: Literal["raw", "jpeg"] = Field(
        default="raw",
        description="画像データ形式 (raw: 生配列 / jpeg: 圧縮)",
    )
    shape: list[int] | None = Field(
        default=None,
        description="numpy 配列の shape (raw 形式時に必須, 例: [480, 640, 3])",
    )
    dtype: str = Field(
        default="uint8",
        description="numpy 配列の dtype (raw 形式時に使用, uint8 のみ)",
    )
    score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "検出信頼度の下限しきい値. config の infer_score_threshold と二段フィルタに"
            "なり, 実際の下限は max(request, config) となる."
        ),
    )

    @model_validator(mode="after")
    def validate_raw_format(self) -> Self:
        """Raw フォーマット時に shape が必須であることを検証する."""
        if self.format == "raw" and self.shape is None:
            raise ValueError("raw フォーマットでは shape が必須です")
        if self.format == "raw" and self.shape is not None and len(self.shape) != 3:
            raise ValueError("shape は [height, width, 3] の形式である必要があります")
        if self.format == "raw" and self.shape is not None and len(self.shape) == 3:
            height, width = self.shape[0], self.shape[1]
            if height * width > MAX_PIXELS:
                raise ValueError(
                    f"画像サイズが上限を超えています: {height}x{width} "
                    f"(上限: {MAX_PIXELS} ピクセル)"
                )
        return self

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        """Dtype が許可されたホワイトリストに含まれることを検証する."""
        if v not in _ALLOWED_DTYPES:
            allowed = ", ".join(sorted(_ALLOWED_DTYPES))
            raise ValueError(
                f"許可されていない dtype: {v}. "
                f"次のいずれかを指定してください: {allowed}"
            )
        return v


class DetectionDict(BaseModel):
    """検出結果 1 件."""

    class_id: int = Field(description="クラス ID")
    class_name: str = Field(description="クラス名")
    confidence: float = Field(ge=0.0, le=1.0, description="信頼度")
    bbox: list[float] = Field(
        description="バウンディングボックス [x1, y1, x2, y2] (元画像座標系, ピクセル)",
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox_length(cls, v: list[float]) -> list[float]:
        """Bbox が 4 要素であることを検証する."""
        if len(v) != 4:
            raise ValueError(
                f"bbox は 4 要素である必要があります. 受け取った: {len(v)}"
            )
        return v


class DetectResponse(BaseModel):
    """検出レスポンス."""

    detections: list[DetectionDict] = Field(description="検出結果のリスト")
    e2e_time_ms: float = Field(description="エンドツーエンド処理時間 (ミリ秒)")
    backend: str = Field(description="使用バックエンド")
    phase_times_ms: dict[str, float] | None = Field(
        default=None,
        description=(
            "フェーズ別タイミング (ms). Pipeline 内の "
            "pipeline_preprocess_ms / pipeline_inference_ms / "
            "pipeline_postprocess_ms に加え, API boundary の "
            "api_preprocess_ms (deserialize + cvtColor) / "
            "api_postprocess_ms (results 組み立て + DetectResponse 構築) も含む. "
            "全キー合計は e2e_time_ms と概ね一致する. "
            "CUDA 利用時は pipeline_inference_gpu_ms (CUDA Event 計測の GPU 実時間) "
            "も追加され, pipeline_inference_ms との差分が Python 側待ち時間 "
            "(GIL / asyncio / OS scheduler) の指標になる."
        ),
    )
    gpu_clock_mhz: int | None = Field(
        default=None,
        description=(
            "現在の GPU graphics clock (MHz). pynvml 初期化失敗 "
            "(CUDA 不在 / NVIDIA driver なし) 時は null."
        ),
    )
    gpu_vram_used_mb: int | None = Field(
        default=None,
        description=("現在の GPU VRAM 使用量 (MB). pynvml 初期化失敗時は null."),
    )
    gpu_temperature_c: int | None = Field(
        default=None,
        description=("現在の GPU 温度 (℃). pynvml 初期化失敗時は null."),
    )
