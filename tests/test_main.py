"""main.pyのテスト."""

import pytest

from main import main


def test_main(capsys: pytest.CaptureFixture[str]) -> None:
    """main関数が正しく出力することを確認する."""
    main()
    captured = capsys.readouterr()
    assert "Hello from pochidetection!" in captured.out
