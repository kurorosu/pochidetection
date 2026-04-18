"""pochi serve サブコマンド.

WebAPI サーバー (FastAPI + uvicorn) を起動する.
"""

import argparse
from pathlib import Path
from typing import Any

import uvicorn

from pochidetection.api import app as app_module
from pochidetection.api.app import build_engine, create_app
from pochidetection.api.config import ServerConfig
from pochidetection.logging import LoggerManager
from pochidetection.logging.logger_manager import COLORLOG_AVAILABLE

logger = LoggerManager().get_logger(__name__)

# uvicorn は %(module) だと全て "h11_impl" 等になるため %(name) を使用する.
_UVICORN_COLOR_FORMAT = (
    "%(asctime)s|%(log_color)s%(levelname)-5.5s%(reset)s|"
    "%(name)-18s|%(lineno)03d| %(message)s"
)
_UVICORN_PLAIN_FORMAT = (
    "%(asctime)s|%(levelname)-5.5s|%(name)-18s|%(lineno)03d| %(message)s"
)
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white",
}


def _build_uvicorn_log_config(log_level: str) -> dict[str, Any]:
    """Uvicorn 用のログ設定を LoggerManager と同一フォーマットで生成する."""
    if COLORLOG_AVAILABLE:
        formatter_config: dict[str, Any] = {
            "()": "colorlog.ColoredFormatter",
            "format": _UVICORN_COLOR_FORMAT,
            "datefmt": _LOG_DATE_FORMAT,
            "log_colors": _LOG_COLORS,
        }
    else:
        formatter_config = {
            "format": _UVICORN_PLAIN_FORMAT,
            "datefmt": _LOG_DATE_FORMAT,
        }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": formatter_config,
            "access": formatter_config,
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": log_level.upper(),
                "propagate": False,
            },
            "uvicorn.error": {
                "level": log_level.upper(),
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": log_level.upper(),
                "propagate": False,
            },
        },
    }


def run_serve(args: argparse.Namespace) -> None:
    """Run the WebAPI server.

    Args:
        args: コマンドライン引数.
    """
    logger.info("=== pochidetection serve mode ===")

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    server_config = ServerConfig(
        model_path=model_path,
        config_path=Path(args.config) if args.config else None,
        host=args.host,
        port=args.port,
        pipeline=args.pipeline,
    )

    # uvicorn 起動前にモデルロード + warmup を実行し, 失敗時はトレースバックを
    # 抑制してクリーンに終了する (lifespan 失敗時の traceback 抑止).
    try:
        engine = build_engine(server_config)
    except Exception as e:  # noqa: BLE001
        logger.error(f"モデルロードに失敗しました: {e}")
        return

    try:
        logger.info("Running warmup inference...")
        engine.warmup()
        logger.info("Warmup complete")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Warmup 推論に失敗しました: {e}")
        engine.close()
        return

    app_module._engine = engine
    app = create_app()

    logger.info(
        f"Starting WebAPI server: http://{server_config.host}:{server_config.port}"
    )

    debug = getattr(args, "debug", False)
    log_level = "debug" if debug else "info"
    uvicorn.run(
        app,
        host=server_config.host,
        port=server_config.port,
        log_level=log_level,
        log_config=_build_uvicorn_log_config(log_level),
    )
