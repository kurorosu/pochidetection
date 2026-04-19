"""GPU graphics clock 取得ユーティリティ (pynvml ラッパ).

`POST /api/v1/detect` のログに現在の GPU クロックを併記し,
NVIDIA driver の adaptive clock policy による推論時間振れを観察するために使用する.

CUDA 不在 / NVIDIA driver なし / pynvml 初期化失敗時は ``None`` を返し,
呼び出し側を阻害しない設計にしている. 初回呼び出し時に lazy init し,
2 回目以降は handle を再利用する.
"""

import pynvml

from pochidetection.logging import LoggerManager

logger = LoggerManager().get_logger(__name__)

_handle: object | None = None
_initialized: bool = False


def get_gpu_clock_mhz() -> int | None:
    """現在の GPU graphics clock (MHz) を返す.

    Returns:
        graphics clock (MHz). 取得不可な環境では ``None``.
    """
    global _handle, _initialized
    if not _initialized:
        _initialized = True
        try:
            pynvml.nvmlInit()
            _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError as e:
            logger.warning(f"pynvml unavailable, GPU clock monitoring disabled: {e}")
            _handle = None

    if _handle is None:
        return None
    try:
        return int(pynvml.nvmlDeviceGetClockInfo(_handle, pynvml.NVML_CLOCK_GRAPHICS))
    except pynvml.NVMLError:
        return None
