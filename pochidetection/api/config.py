"""API サーバー設定."""

from pathlib import Path

from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """推論 API サーバーの起動設定."""

    model_path: Path = Field(
        description="学習済みモデルのパス (ディレクトリ / .onnx / .engine)"
    )
    config_path: Path | None = Field(
        default=None,
        description="pochidetection 設定ファイルパス. 未指定時はモデルパスから自動検出",
    )
    host: str = Field(default="127.0.0.1", description="バインドホスト")
    port: int = Field(default=8000, description="バインドポート")
