"""WebAPI テスト共通 fixture."""

from collections.abc import Iterator

import pytest

from pochidetection.api import state as state_module


@pytest.fixture(autouse=True)
def reset_engine() -> Iterator[None]:
    """各テスト境界で ``state._engine`` を元の値に復元する.

    Why: 複数テストがグローバル ``_engine`` を直接書き換えるため, teardown で
    自動復元しないと実行順序や pytest-xdist で state が残留する.
    """
    original = state_module._engine
    try:
        yield
    finally:
        state_module.set_engine(original)
