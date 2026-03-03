"""RT-DETR CLI のテスト."""

import subprocess
import sys


class TestCliNoCommand:
    """サブコマンド未指定時の動作テスト."""

    def test_no_command_shows_help_without_error(self) -> None:
        """サブコマンドなしで実行するとヘルプが表示され正常終了することを確認."""
        result = subprocess.run(
            [sys.executable, "-m", "pochidetection.cli.rtdetr"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "使用例" in result.stdout
