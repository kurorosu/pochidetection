"""`pochidetection.api.gpu_metrics` の 3 関数の分岐を検証.

pynvml の各失敗経路 (init / handle 取得 / メトリクス取得) で ``None`` を返すこと,
および handle のキャッシュ挙動 (2 回目以降 init/handle 取得を再実行せず, 3 関数で
共有される) を確認する.

pynvml の import 自体は `gpu_metrics` モジュール top-level で行われるため,
"pynvml 未インストール" 経路は `nvmlInit()` の `NVMLError` として再現する.
"""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pynvml
import pytest
import torch

from pochidetection.api import gpu_metrics


@pytest.fixture(autouse=True)
def _reset_gpu_metrics_state() -> Iterator[None]:
    """各テストの前後でモジュール level のキャッシュ状態を初期化する."""
    gpu_metrics._handle = None
    gpu_metrics._initialized = False
    yield
    gpu_metrics._handle = None
    gpu_metrics._initialized = False


# ---------------------------------------------------------------------------
# get_gpu_clock_mhz
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failing_call",
    ["init", "handle", "clock"],
    ids=[
        "nvmlInit_fails",
        "nvmlDeviceGetHandleByIndex_fails",
        "nvmlDeviceGetClockInfo_fails",
    ],
)
def test_clock_returns_none_on_nvml_failure(failing_call: str) -> None:
    """pynvml の各呼び出し失敗時に ``None`` を返す."""
    fake_handle = object()
    err = pynvml.NVMLError(1)
    init_effect = err if failing_call == "init" else None
    handle_effect = err if failing_call == "handle" else None
    handle_return = None if failing_call == "handle" else fake_handle
    clock_effect = err if failing_call == "clock" else None
    clock_return = None if failing_call == "clock" else 1650

    with (
        patch.object(pynvml, "nvmlInit", side_effect=init_effect, return_value=None),
        patch.object(
            pynvml,
            "nvmlDeviceGetHandleByIndex",
            side_effect=handle_effect,
            return_value=handle_return,
        ),
        patch.object(
            pynvml,
            "nvmlDeviceGetClockInfo",
            side_effect=clock_effect,
            return_value=clock_return,
        ),
    ):
        assert gpu_metrics.get_gpu_clock_mhz() is None


def test_clock_returns_integer_mhz_on_success_with_mock() -> None:
    """正常系: `nvmlDeviceGetClockInfo()` の戻り値を int で返す (mock 版)."""
    fake_handle = object()
    with (
        patch.object(pynvml, "nvmlInit", return_value=None),
        patch.object(pynvml, "nvmlDeviceGetHandleByIndex", return_value=fake_handle),
        patch.object(pynvml, "nvmlDeviceGetClockInfo", return_value=1650),
    ):
        result = gpu_metrics.get_gpu_clock_mhz()

    assert result == 1650
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# get_gpu_vram_used_mb
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failing_call",
    ["init", "handle", "memory"],
    ids=[
        "nvmlInit_fails",
        "nvmlDeviceGetHandleByIndex_fails",
        "nvmlDeviceGetMemoryInfo_fails",
    ],
)
def test_vram_returns_none_on_nvml_failure(failing_call: str) -> None:
    """VRAM 取得で各呼び出し失敗時に ``None`` を返す."""
    fake_handle = object()
    fake_info = SimpleNamespace(used=1024 * 1024 * 512)  # 512 MiB
    err = pynvml.NVMLError(1)
    init_effect = err if failing_call == "init" else None
    handle_effect = err if failing_call == "handle" else None
    handle_return = None if failing_call == "handle" else fake_handle
    mem_effect = err if failing_call == "memory" else None
    mem_return = None if failing_call == "memory" else fake_info

    with (
        patch.object(pynvml, "nvmlInit", side_effect=init_effect, return_value=None),
        patch.object(
            pynvml,
            "nvmlDeviceGetHandleByIndex",
            side_effect=handle_effect,
            return_value=handle_return,
        ),
        patch.object(
            pynvml,
            "nvmlDeviceGetMemoryInfo",
            side_effect=mem_effect,
            return_value=mem_return,
        ),
    ):
        assert gpu_metrics.get_gpu_vram_used_mb() is None


def test_vram_returns_integer_mb_on_success_with_mock() -> None:
    """正常系: `nvmlDeviceGetMemoryInfo().used` を MB 換算した整数で返す."""
    fake_handle = object()
    fake_info = SimpleNamespace(used=1024 * 1024 * 2048)  # 2048 MiB
    with (
        patch.object(pynvml, "nvmlInit", return_value=None),
        patch.object(pynvml, "nvmlDeviceGetHandleByIndex", return_value=fake_handle),
        patch.object(pynvml, "nvmlDeviceGetMemoryInfo", return_value=fake_info),
    ):
        result = gpu_metrics.get_gpu_vram_used_mb()

    assert result == 2048
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# get_gpu_temperature_c
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failing_call",
    ["init", "handle", "temperature"],
    ids=[
        "nvmlInit_fails",
        "nvmlDeviceGetHandleByIndex_fails",
        "nvmlDeviceGetTemperature_fails",
    ],
)
def test_temperature_returns_none_on_nvml_failure(failing_call: str) -> None:
    """温度取得で各呼び出し失敗時に ``None`` を返す."""
    fake_handle = object()
    err = pynvml.NVMLError(1)
    init_effect = err if failing_call == "init" else None
    handle_effect = err if failing_call == "handle" else None
    handle_return = None if failing_call == "handle" else fake_handle
    temp_effect = err if failing_call == "temperature" else None
    temp_return = None if failing_call == "temperature" else 55

    with (
        patch.object(pynvml, "nvmlInit", side_effect=init_effect, return_value=None),
        patch.object(
            pynvml,
            "nvmlDeviceGetHandleByIndex",
            side_effect=handle_effect,
            return_value=handle_return,
        ),
        patch.object(
            pynvml,
            "nvmlDeviceGetTemperature",
            side_effect=temp_effect,
            return_value=temp_return,
        ),
    ):
        assert gpu_metrics.get_gpu_temperature_c() is None


def test_temperature_returns_integer_celsius_on_success_with_mock() -> None:
    """正常系: `nvmlDeviceGetTemperature()` の戻り値を int で返す."""
    fake_handle = object()
    with (
        patch.object(pynvml, "nvmlInit", return_value=None),
        patch.object(pynvml, "nvmlDeviceGetHandleByIndex", return_value=fake_handle),
        patch.object(pynvml, "nvmlDeviceGetTemperature", return_value=62),
    ):
        result = gpu_metrics.get_gpu_temperature_c()

    assert result == 62
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# handle キャッシュの共有挙動
# ---------------------------------------------------------------------------


def test_handle_is_cached_across_calls() -> None:
    """2 回目以降は init / handle 取得を再実行せず, メトリクス取得のみ行う."""
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
        first = gpu_metrics.get_gpu_clock_mhz()
        second = gpu_metrics.get_gpu_clock_mhz()

    assert first == 1500
    assert second == 1500
    assert mock_init.call_count == 1
    assert mock_get_handle.call_count == 1
    # メトリクス取得自体は毎回実行される (キャッシュ対象外).
    assert mock_get_clock.call_count == 2


def test_handle_is_shared_across_metrics() -> None:
    """3 関数で handle を共有する (init / handle 取得は 1 回のみ)."""
    fake_handle = object()
    fake_info = SimpleNamespace(used=1024 * 1024 * 1024)  # 1024 MiB
    with (
        patch.object(pynvml, "nvmlInit", return_value=None) as mock_init,
        patch.object(
            pynvml, "nvmlDeviceGetHandleByIndex", return_value=fake_handle
        ) as mock_get_handle,
        patch.object(pynvml, "nvmlDeviceGetClockInfo", return_value=1500),
        patch.object(pynvml, "nvmlDeviceGetMemoryInfo", return_value=fake_info),
        patch.object(pynvml, "nvmlDeviceGetTemperature", return_value=55),
    ):
        assert gpu_metrics.get_gpu_clock_mhz() == 1500
        assert gpu_metrics.get_gpu_vram_used_mb() == 1024
        assert gpu_metrics.get_gpu_temperature_c() == 55

    assert mock_init.call_count == 1
    assert mock_get_handle.call_count == 1


def test_handle_stays_none_after_init_failure() -> None:
    """init 失敗後は 2 回目以降も再 init せず即 ``None`` を返す (全関数共通)."""
    with (
        patch.object(pynvml, "nvmlInit", side_effect=pynvml.NVMLError(1)) as mock_init,
        patch.object(pynvml, "nvmlDeviceGetHandleByIndex") as mock_get_handle,
    ):
        assert gpu_metrics.get_gpu_clock_mhz() is None
        assert gpu_metrics.get_gpu_vram_used_mb() is None
        assert gpu_metrics.get_gpu_temperature_c() is None

    assert mock_init.call_count == 1
    assert mock_get_handle.call_count == 0


# ---------------------------------------------------------------------------
# 実 CUDA 環境での統合
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_returns_real_clock_mhz_on_cuda_environment() -> None:
    """CUDA 環境では実際の GPU graphics clock (MHz) を整数で返す."""
    result = gpu_metrics.get_gpu_clock_mhz()

    assert result is not None
    assert isinstance(result, int)
    assert result > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_returns_real_vram_used_mb_on_cuda_environment() -> None:
    """CUDA 環境では実際の VRAM 使用量 (MB) を整数で返す."""
    result = gpu_metrics.get_gpu_vram_used_mb()

    assert result is not None
    assert isinstance(result, int)
    assert result >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_returns_real_temperature_c_on_cuda_environment() -> None:
    """CUDA 環境では実際の GPU 温度 (℃) を整数で返す."""
    result = gpu_metrics.get_gpu_temperature_c()

    assert result is not None
    assert isinstance(result, int)
    assert 0 < result < 150
