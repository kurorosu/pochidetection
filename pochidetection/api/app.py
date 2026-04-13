"""FastAPI アプリケーション."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI

from pochidetection.api.config import ServerConfig
from pochidetection.api.routers import health
from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.logging import LoggerManager
from pochidetection.scripts.common.inference import (
    is_onnx_model,
    is_tensorrt_model,
    resolve_and_setup_pipeline,
)
from pochidetection.utils.config_loader import ConfigLoader
from pochidetection.utils.config_resolver import resolve_config_path

logger = LoggerManager().get_logger(__name__)

DEFAULT_CONFIG = "configs/rtdetr_coco.py"


@dataclass(frozen=True)
class ModelHolder:
    """ロード済みモデルとメタ情報を保持する.

    Issue #436 で IDetectionBackend に差し替える前提の薄いホルダ.
    """

    pipeline: IDetectionPipeline  # type: ignore[type-arg]
    config: DetectionConfigDict
    architecture: str
    class_names: list[str]
    num_classes: int
    input_size: tuple[int, int]
    model_path: str
    backend_name: str


_holder: ModelHolder | None = None


def get_holder() -> ModelHolder:
    """グローバルモデルホルダを取得する.

    Raises:
        RuntimeError: モデルが初期化されていない場合.
    """
    if _holder is None:
        raise RuntimeError("モデルが初期化されていません")
    return _holder


def _detect_backend_name(model_path: Path) -> str:
    """モデルパスからバックエンド名を判定する."""
    if is_tensorrt_model(model_path):
        return "tensorrt"
    if is_onnx_model(model_path):
        return "onnx"
    return "pytorch"


def _build_holder(server_config: ServerConfig) -> ModelHolder:
    """設定からモデルをロードし ModelHolder を構築する."""
    config_path = resolve_config_path(
        config=str(server_config.config_path) if server_config.config_path else None,
        model_dir=str(server_config.model_path),
        default_config=DEFAULT_CONFIG,
    )
    logger.info(f"Loading config: {config_path}")

    config = ConfigLoader.load(config_path)

    logger.info(f"Loading model: {server_config.model_path}")
    resolved = resolve_and_setup_pipeline(
        config=config,
        model_dir=str(server_config.model_path),
        config_path=config_path,
    )
    if resolved is None:
        raise RuntimeError(f"モデルロードに失敗しました: {server_config.model_path}")

    image_size = config["image_size"]
    height, width = image_size["height"], image_size["width"]

    class_names = config.get("class_names") or []
    backend_name = _detect_backend_name(server_config.model_path)

    return ModelHolder(
        pipeline=resolved.ctx.pipeline,
        config=config,
        architecture=config["architecture"],
        class_names=list(class_names),
        num_classes=config["num_classes"],
        input_size=(height, width),
        model_path=str(resolved.model_path),
        backend_name=backend_name,
    )


def _warmup(holder: ModelHolder) -> None:
    """ダミー画像で 1 回推論し, モデルをウォームアップする."""
    height, width = holder.input_size
    dummy = np.zeros((height, width, 3), dtype=np.uint8)
    holder.pipeline.run(dummy)


def _create_lifespan(
    server_config: ServerConfig,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Lifespan コンテキストマネージャを生成する."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        global _holder

        _holder = _build_holder(server_config)
        logger.info(
            f"Model loaded: architecture={_holder.architecture}, "
            f"backend={_holder.backend_name}, num_classes={_holder.num_classes}"
        )

        logger.info("Running warmup inference...")
        _warmup(_holder)
        logger.info("Warmup complete")

        yield

        _holder = None
        logger.info("Server shutdown complete")

    return lifespan


def create_app(server_config: ServerConfig | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        server_config: サーバー設定. None を渡すと lifespan なしで生成 (テスト用).

    Returns:
        FastAPI アプリケーション.
    """
    lifespan = _create_lifespan(server_config) if server_config else None

    app = FastAPI(
        title="pochidetection Inference API",
        version="1.0.0",
        description="pochidetection 検出推論 API サーバー",
        lifespan=lifespan,
    )

    app.include_router(health.router)

    return app
