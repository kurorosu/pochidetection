"""`pochidetection.api.gpu_clock.get_gpu_clock_mhz` の分岐を検証.

pynvml の各失敗経路 (init / handle 取得 / clock 取得) で ``None`` を返すこと,
および handle のキャッシュ挙動 (2 回目以降 init/handle 取得を再実行しない) を確認する.

pynvml の import 自体は `gpu_clock` モジュール top-level で行われるため,
"pynvml 未インストール" 経路は `nvmlInit()` の `NVMLError` として再現する.
"""

from collections.abc import Iterator
from unittest.mock import patch

import pynvml
import pytest
import torch

from pochidetection.api import gpu_clock


@pytest.fixture(autouse=True)
def _reset_gpu_clock_state() -> Iterator[None]:
    """各テストの前後でモジュール level のキャッシュ状態を初期化する."""
    gpu_clock._handle = None
    gpu_clock._initialized = False
    yield
    gpu_clock._handle = None
    gpu_clock._initialized = False


def test_returns_none_when_nvml_init_fails() -> None:
    """`nvmlInit()` が NVMLError を送出した場合 ``None`` を返す.

    pynvml 未インストール時も NVIDIA driver なしで `nvmlInit()` が失敗するため,
    同一経路でカバーできる.
    """
    with patch.object(pynvml, "nvmlInit", side_effect=pynvml.NVMLError(1)):
        assert gpu_clock.get_gpu_clock_mhz() is None


def test_returns_none_when_handle_acquisition_fails() -> None:
    """`nvmlDeviceGetHandleByIndex()` 失敗時 ``None`` を返す."""
    with (
        patch.object(pynvml, "nvmlInit", return_value=None),
        patch.object(
            pynvml,
            "nvmlDeviceGetHandleByIndex",
            side_effect=pynvml.NVMLError(1),
        ),
    ):
        assert gpu_clock.get_gpu_clock_mhz() is None


def test_returns_none_when_clock_info_fails() -> None:
    """`nvmlDeviceGetClockInfo()` 失敗時 ``None`` を返す."""
    fake_handle = object()
    with (
        patch.object(pynvml, "nvmlInit", return_value=None),
        patch.object(pynvml, "nvmlDeviceGetHandleByIndex", return_value=fake_handle),
        patch.object(
            pynvml,
            "nvmlDeviceGetClockInfo",
            side_effect=pynvml.NVMLError(1),
        ),
    ):
        assert gpu_clock.get_gpu_clock_mhz() is None


def test_returns_integer_mhz_on_success_with_mock() -> None:
    """正常系: `nvmlDeviceGetClockInfo()` の戻り値を int で返す (mock 版)."""
    fake_handle = object()
    with (
        patch.object(pynvml, "nvmlInit", return_value=None),
        patch.object(pynvml, "nvmlDeviceGetHandleByIndex", return_value=fake_handle),
        patch.object(pynvml, "nvmlDeviceGetClockInfo", return_value=1650),
    ):
        result = gpu_clock.get_gpu_clock_mhz()

    assert result == 1650
    assert isinstance(result, int)


def test_handle_is_cached_across_calls() -> None:
    """2 回目以降は init / handle 取得を再実行せず clock 取得のみ行う."""
    fake_handle = object()
    with (
        patch.object(pynvml, "nvmlInit", return_value=None) as mock_init,
        patch.object(
            pynvml, "nvmlDeviceGetHandleByIndex", return_value=fake_handle
        ) as mock_get_handle,
        patch.object(
            pynvml, "nvmlDeviceGetClockInfo", return_value=1500
        ) as mock_get_clock,
    ):
        first = gpu_clock.get_gpu_clock_mhz()
        second = gpu_clock.get_gpu_clock_mhz()

    assert first == 1500
    assert second == 1500
    assert mock_init.call_count == 1
    assert mock_get_handle.call_count == 1
    # clock 取得自体は毎回実行される (キャッシュ対象外).
    assert mock_get_clock.call_count == 2


def test_handle_stays_none_after_init_failure() -> None:
    """init 失敗後は 2 回目以降も再 init せず即 ``None`` を返す."""
    with (
        patch.object(pynvml, "nvmlInit", side_effect=pynvml.NVMLError(1)) as mock_init,
        patch.object(pynvml, "nvmlDeviceGetHandleByIndex") as mock_get_handle,
    ):
        assert gpu_clock.get_gpu_clock_mhz() is None
        assert gpu_clock.get_gpu_clock_mhz() is None

    assert mock_init.call_count == 1
    assert mock_get_handle.call_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_returns_real_clock_mhz_on_cuda_environment() -> None:
    """CUDA 環境では実際の GPU graphics clock (MHz) を整数で返す."""
    result = gpu_clock.get_gpu_clock_mhz()

    assert result is not None
    assert isinstance(result, int)
    assert result > 0
