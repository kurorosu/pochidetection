"""FastAPI アプリケーション."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from pochidetection.api.backends import IDetectionBackend, create_detection_backend
from pochidetection.api.config import ServerConfig
from pochidetection.api.routers import health, inference
from pochidetection.logging import LoggerManager
from pochidetection.utils.config_loader import ConfigLoader
from pochidetection.utils.config_resolver import resolve_config_path

logger = LoggerManager().get_logger(__name__)

DEFAULT_CONFIG = "configs/rtdetr_coco.py"

_engine: IDetectionBackend | None = None


def get_engine() -> IDetectionBackend:
    """Return the global detection backend.

    Raises:
        RuntimeError: バックエンドが初期化されていない場合.
    """
    if _engine is None:
        raise RuntimeError("モデルが初期化されていません")
    return _engine


def build_engine(server_config: ServerConfig) -> IDetectionBackend:
    """Load config and build the backend for the given server config."""
    config_path = resolve_config_path(
        config=str(server_config.config_path) if server_config.config_path else None,
        model_dir=str(server_config.model_path),
        default_config=DEFAULT_CONFIG,
    )
    logger.info(f"Loading config: {config_path}")

    config = ConfigLoader.load(config_path)
    return create_detection_backend(
        model_path=server_config.model_path,
        config=config,
        config_path=config_path,
    )


def _create_lifespan(
    server_config: ServerConfig,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Create the lifespan context manager bound to the server config."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        global _engine

        _engine = build_engine(server_config)
        info = _engine.get_model_info()
        logger.info(
            f"Model loaded: architecture={info['architecture']}, "
            f"backend={_engine.backend_name}, num_classes={info['num_classes']}"
        )

        logger.info("Running warmup inference...")
        _engine.warmup()
        logger.info("Warmup complete")

        yield

        if _engine is not None:
            _engine.close()
        _engine = None
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
    app.include_router(inference.router)

    return app
