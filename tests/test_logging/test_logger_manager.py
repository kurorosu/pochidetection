"""LoggerManagerのテスト."""

import logging

from pochidetection.logging import LoggerManager, LogLevel


class TestLoggerManager:
    """LoggerManagerのテスト."""

    def test_singleton(self) -> None:
        """シングルトンパターンを確認."""
        manager1 = LoggerManager()
        manager2 = LoggerManager()
        assert manager1 is manager2

    def test_get_logger(self) -> None:
        """ロガーを取得できることを確認."""
        manager = LoggerManager()
        logger = manager.get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_get_same_logger(self) -> None:
        """同じ名前のロガーは同じインスタンスを返すことを確認."""
        manager = LoggerManager()
        logger1 = manager.get_logger("same_name")
        logger2 = manager.get_logger("same_name")
        assert logger1 is logger2

    def test_get_logger_with_level(self) -> None:
        """ログレベルを指定してロガーを取得できることを確認."""
        manager = LoggerManager()
        logger = manager.get_logger("debug_logger", LogLevel.DEBUG)
        assert logger.level == logging.DEBUG

    def test_set_level(self) -> None:
        """ログレベルを変更できることを確認."""
        manager = LoggerManager()
        logger = manager.get_logger("level_test")
        manager.set_level("level_test", LogLevel.WARNING)
        assert logger.level == logging.WARNING

    def test_set_default_level(self) -> None:
        """デフォルトログレベルを変更できることを確認."""
        manager = LoggerManager()
        manager.set_default_level(LogLevel.DEBUG)
        # 新規ロガーがデフォルトレベル (DEBUG) で作成されることを検証
        logger = manager.get_logger("test_default_level_check")
        assert logger.level == logging.DEBUG
        # 元に戻す
        manager.set_default_level(LogLevel.INFO)

    def test_preserves_external_handler(self) -> None:
        """外部が追加した handler を破壊しないことを確認.

        pytest caplog 等が logger に付けた handler を ``handlers.clear()`` で
        消していた旧挙動のリグレッション防止.
        """
        name = "external_handler_target"
        external = logging.StreamHandler()
        logging.getLogger(name).addHandler(external)

        manager = LoggerManager()
        logger = manager.get_logger(name)

        assert external in logger.handlers

    def test_does_not_duplicate_owned_handler(self) -> None:
        """自前 handler (``_pochi_owned``) が重複追加されないことを確認."""
        name = "owned_handler_check"
        manager = LoggerManager()
        logger = manager.get_logger(name)
        owned_count_first = sum(
            1 for h in logger.handlers if getattr(h, "_pochi_owned", False)
        )

        # 内部 _create_logger を再度通しても自前 handler は 1 つのまま.
        manager._create_logger(name, LogLevel.INFO)
        owned_count_second = sum(
            1 for h in logger.handlers if getattr(h, "_pochi_owned", False)
        )

        assert owned_count_first == 1
        assert owned_count_second == 1
