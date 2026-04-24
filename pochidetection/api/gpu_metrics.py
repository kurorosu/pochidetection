"""GPU メトリクス取得ユーティリティ (pynvml ラッパ).

`POST /api/v1/detect` のレスポンスに GPU の graphics clock / VRAM 使用量 / 温度 を
併記し, クライアント側で NVIDIA adaptive clock policy によるクロック変動や VRAM
圧迫 / サーマルスロットリングと推論時間の振れを相関分析するために使用する.

CUDA 不在 / NVIDIA driver なし / pynvml 初期化失敗時は各関数とも ``None`` を返し,
呼び出し側を阻害しない設計にしている. 初回呼び出し時に lazy init し, 2 回目以降は
handle を再利用する. handle は 3 関数で共有する.
"""

import pynvml

from pochidetection.logging import LoggerManager

__all__ = [
    "get_gpu_clock_mhz",
    "get_gpu_temperature_c",
    "get_gpu_vram_used_mb",
]

logger = LoggerManager().get_logger(__name__)

_handle: object | None = None
_initialized: bool = False


def _get_handle() -> object | None:
    """Pynvml handle を lazy init して返す (キャッシュ共有)."""
    global _handle, _initialized
    if not _initialized:
        _initialized = True
        try:
            pynvml.nvmlInit()
            _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError as e:
            logger.warning(f"pynvml unavailable, GPU metrics disabled: {e}")
            _handle = None
    return _handle


def get_gpu_clock_mhz() -> int | None:
    """現在の GPU graphics clock (MHz) を返す.

    Returns:
        graphics clock (MHz). 取得不可な環境では ``None``.
    """
    handle = _get_handle()
    if handle is None:
        return None
    try:
        return int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
    except pynvml.NVMLError:
        return None


def get_gpu_vram_used_mb() -> int | None:
    """現在の GPU VRAM 使用量 (MB) を返す.

    Returns:
        VRAM 使用量 (MB). 取得不可な環境では ``None``.
    """
    handle = _get_handle()
    if handle is None:
        return None
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used // (1024 * 1024))
    except pynvml.NVMLError:
        return None


def get_gpu_temperature_c() -> int | None:
    """現在の GPU 温度 (℃) を返す.

    Returns:
        GPU 温度 (℃). 取得不可な環境では ``None``.
    """
    handle = _get_handle()
    if handle is None:
        return None
    try:
        return int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except pynvml.NVMLError:
        return None
