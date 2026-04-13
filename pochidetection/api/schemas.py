"""API レスポンススキーマ."""

from pydantic import BaseModel, Field


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
