"""ログ管理マネージャー.

Singletonパターンによるアプリケーション全体のログ管理.
"""

import logging
from enum import Enum
from typing import Optional

try:
    import colorlog

    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


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
        _loggers: 管理されているロガーの辞書.
        _default_level: デフォルトのログレベル.
        _format_string: ログフォーマット文字列.
    """

    _instance: Optional["LoggerManager"] = None
    _loggers: dict[str, logging.Logger] = {}

    def __new__(cls) -> "LoggerManager":
        """シングルトンパターンの実装."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """LoggerManagerを初期化."""
        if hasattr(self, "_initialized"):
            return

        self._default_level = LogLevel.INFO
        self._format_string = (
            "[%(asctime)s][%(log_color)s%(levelname)s%(reset)s]"
            "[%(name)s][%(filename)s:%(lineno)d] %(message)s"
        )
        self._plain_format_string = (
            "[%(asctime)s][%(levelname)s]"
            "[%(name)s][%(filename)s:%(lineno)d] %(message)s"
        )
        self._date_format = "%Y-%m-%d %H:%M:%S"
        self._log_colors = {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        }
        self._initialized = True

    def get_logger(self, name: str, level: Optional[LogLevel] = None) -> logging.Logger:
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

        Args:
            name: ロガー名.
            level: ログレベル.

        Returns:
            設定されたロガー.
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.value))

        # 既存のハンドラーをクリア
        logger.handlers.clear()

        # ハンドラーを追加
        handler = self._create_handler()
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

        Args:
            level: 新しいデフォルトログレベル.
        """
        self._default_level = level
