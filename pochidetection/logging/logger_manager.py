"""ログ管理マネージャー.

Singletonパターンによるアプリケーション全体のログ管理.
"""

from __future__ import annotations

import logging
from enum import Enum

try:
    import colorlog

    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
"""ログ出力の日付フォーマット. LoggerManager / uvicorn 共通."""

LOG_COLORS: dict[str, str] = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARN": "yellow",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white",
}
"""colorlog 用のレベル別色設定. LoggerManager / uvicorn 共通."""


class LogLevel(Enum):
    """ログレベル列挙型."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggerManager:
    """ログ管理マネージャークラス.

    Singletonパターンでアプリケーション全体で一貫したログ設定を提供.

    Attributes:
        _default_level: デフォルトのログレベル.
        _format_string: ログフォーマット文字列.
    """

    _instance: LoggerManager | None = None

    def __new__(cls) -> LoggerManager:
        """シングルトンパターンの実装."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """LoggerManagerを初期化."""
        if hasattr(self, "_initialized"):
            return

        self._loggers: dict[str, logging.Logger] = {}
        self._default_level = LogLevel.INFO
        self._format_string = (
            "%(asctime)s|%(log_color)s%(levelname)-5.5s%(reset)s|"
            "%(module)-18s|%(lineno)03d| %(message)s"
        )
        self._plain_format_string = (
            "%(asctime)s|%(levelname)-5.5s|" "%(module)-18s|%(lineno)03d| %(message)s"
        )
        self._date_format = LOG_DATE_FORMAT
        self._log_colors = LOG_COLORS
        self._initialized = True

    def get_logger(self, name: str, level: LogLevel | None = None) -> logging.Logger:
        """指定された名前のロガーを取得または作成.

        Args:
            name: ロガー名.
            level: ログレベル. Noneの場合はデフォルトレベルを使用.

        Returns:
            設定されたロガー.

        Examples:
            >>> manager = LoggerManager()
            >>> logger = manager.get_logger("pochidetection")
            >>> logger.info("ログメッセージ")
        """
        if name in self._loggers:
            return self._loggers[name]

        logger = self._create_logger(name, level or self._default_level)
        self._loggers[name] = logger
        return logger

    def _create_logger(self, name: str, level: LogLevel) -> logging.Logger:
        """新しいロガーを作成.

        Why: ``logging.getLogger(name)`` は同名ならプロセス全体で同一インスタンスを
        返すため, ここで ``handlers.clear()`` を呼ぶと pytest caplog など外部が
        追加した handler も破壊してしまう. 自前 handler には ``_pochi_owned``
        マーカを付け, 既に存在する場合は重複追加せず, 外部 handler は残す.

        Args:
            name: ロガー名.
            level: ログレベル.

        Returns:
            設定されたロガー.
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.value))

        if not any(getattr(h, "_pochi_owned", False) for h in logger.handlers):
            handler = self._create_handler()
            setattr(handler, "_pochi_owned", True)
            logger.addHandler(handler)

        # 親ロガーへの伝播を無効化
        logger.propagate = False

        return logger

    def _create_handler(self) -> logging.Handler:
        """ハンドラーを作成.

        Returns:
            設定されたハンドラー.
        """
        handler = logging.StreamHandler()

        if COLORLOG_AVAILABLE:
            formatter = colorlog.ColoredFormatter(
                self._format_string,
                datefmt=self._date_format,
                log_colors=self._log_colors,
            )
        else:
            formatter = logging.Formatter(
                self._plain_format_string,
                datefmt=self._date_format,
            )

        handler.setFormatter(formatter)
        return handler

    def set_level(self, name: str, level: LogLevel) -> None:
        """指定されたロガーのログレベルを変更.

        Args:
            name: ロガー名.
            level: 新しいログレベル.
        """
        if name in self._loggers:
            self._loggers[name].setLevel(getattr(logging, level.value))

    def set_default_level(self, level: LogLevel) -> None:
        """デフォルトのログレベルを変更.

        既存の全ロガーのレベルも同時に更新する.

        Args:
            level: 新しいデフォルトログレベル.
        """
        self._default_level = level
        self._update_existing_loggers_level()

    def _update_existing_loggers_level(self) -> None:
        """既存ロガーのレベル設定を更新する."""
        log_level = getattr(logging, self._default_level.value)
        for logger in self._loggers.values():
            logger.setLevel(log_level)
