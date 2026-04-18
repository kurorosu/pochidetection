"""API サーバー設定."""

from pathlib import Path
from typing import Literal

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
    pipeline: Literal["cpu", "gpu"] | None = Field(
        default=None,
        description=(
            "preprocess の経路. None の場合は backend 種別から自動解決. "
            "CLI --pipeline 指定値が config の pipeline_mode を上書きする."
        ),
    )
