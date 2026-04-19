"""FastAPI アプリケーション."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from pochidetection.api.backends import IDetectionBackend, create_detection_backend
from pochidetection.api.config import ServerConfig
from pochidetection.api.routers import health, inference
from pochidetection.api.state import get_engine, set_engine
from pochidetection.logging import LoggerManager
from pochidetection.utils.config_loader import ConfigLoader
from pochidetection.utils.config_resolver import resolve_config_path

logger = LoggerManager().get_logger(__name__)

DEFAULT_CONFIG = "configs/rtdetr_coco.py"


def build_engine(server_config: ServerConfig) -> IDetectionBackend:
    """Load config and build the backend for the given server config."""
    config_path = resolve_config_path(
        config=str(server_config.config_path) if server_config.config_path else None,
        model_dir=str(server_config.model_path),
        default_config=DEFAULT_CONFIG,
    )
    logger.info(f"Loading config: {config_path}")

    config = ConfigLoader.load(config_path)
    # Why: CLI --pipeline 指定値で config の pipeline_mode を上書き. None なら触らず
    # config 値 (or default None) を維持し, 後段の resolve で backend 種別から決定.
    if server_config.pipeline is not None:
        config["pipeline_mode"] = server_config.pipeline
    return create_detection_backend(
        model_path=server_config.model_path,
        config=config,
        config_path=config_path,
    )


def _create_lifespan(
    server_config: ServerConfig,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Create the lifespan context manager bound to the server config.

    Note:
        `pochi serve` CLI は ``run_serve`` 側で uvicorn 起動前にモデルロードを済ませ,
        ``create_app()`` (lifespan なし) を使用する. 本 lifespan はプログラマティック
        起動 (`create_app(server_config)`) とテスト用途のために残している.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = build_engine(server_config)
        set_engine(engine)
        info = engine.get_model_info()
        logger.info(
            f"Model loaded: architecture={info['architecture']}, "
            f"backend={engine.backend_name}, num_classes={info['num_classes']}"
        )

        logger.info("Running warmup inference...")
        engine.warmup()
        logger.info("Warmup complete")

        yield

        try:
            current = get_engine()
        except RuntimeError:
            current = None
        if current is not None:
            current.close()
        set_engine(None)
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
