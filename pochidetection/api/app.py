"""FastAPI アプリケーション."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from pochidetection.api.backends import IDetectionBackend, create_detection_backend
from pochidetection.api.config import ServerConfig
from pochidetection.api.constants import MAX_BODY_SIZE
from pochidetection.api.routers import health, inference
from pochidetection.api.state import get_engine, set_engine
from pochidetection.logging import LoggerManager
from pochidetection.utils.config_loader import ConfigLoader
from pochidetection.utils.config_resolver import resolve_config_path

logger = LoggerManager().get_logger(__name__)

DEFAULT_CONFIG = "configs/rtdetr_coco.py"


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """HTTP リクエスト body サイズ上限を強制する middleware.

    ``Content-Length`` ヘッダーを最優先でチェックし, 上限超過なら 413 を返す.
    ヘッダーが不正 (非整数 / 負値) の場合も 413 で弾く. ヘッダーが存在しない
    chunked transfer などは streaming 読み込み時に累積バイト数で判定する.
    """

    def __init__(self, app: FastAPI, max_body_size: int) -> None:
        """Initialize with the wrapped app and the byte limit."""
        super().__init__(app)
        self.max_body_size = max_body_size

    def _too_large_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={
                "detail": (
                    f"Request body too large (limit: {self.max_body_size} bytes)"
                )
            },
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Inspect Content-Length / stream size and short-circuit with 413 if over."""
        content_length_header = request.headers.get("content-length")
        if content_length_header is not None:
            try:
                content_length = int(content_length_header)
            except ValueError:
                return self._too_large_response()
            if content_length < 0 or content_length > self.max_body_size:
                return self._too_large_response()
            return await call_next(request)

        # Content-Length なし (chunked 等) : streaming 読み込み時の累積で判定する.
        body_chunks: list[bytes] = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > self.max_body_size:
                return self._too_large_response()
            body_chunks.append(chunk)

        # 読み切った body を下流で再利用できるよう receive を差し替える.
        cached_body = b"".join(body_chunks)
        more_body_sent = False

        async def receive() -> dict[str, object]:
            nonlocal more_body_sent
            if more_body_sent:
                return {"type": "http.disconnect"}
            more_body_sent = True
            return {
                "type": "http.request",
                "body": cached_body,
                "more_body": False,
            }

        request = Request(request.scope, receive)
        return await call_next(request)


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

    # body サイズ上限 middleware (413 Payload Too Large).
    app.add_middleware(BodySizeLimitMiddleware, max_body_size=MAX_BODY_SIZE)

    app.include_router(health.router)
    app.include_router(inference.router)

    return app
