"""推論時間の計測・集計クラス."""

import time

import torch


class InferenceTimer:
    """推論時間の計測・集計.

    GPU 使用時は CUDA イベントで正確な GPU 時間を計測.
    CPU 使用時は time.perf_counter() を使用.

    Attributes:
        _use_cuda: CUDA を使用するかどうか.
        _total_time_ms: 合計時間 (ミリ秒).
        _count: 計測回数.
        _skip_first: 最初の計測をスキップするかどうか.
    """

    def __init__(self, device: str = "cuda", skip_first: bool = True) -> None:
        """初期化.

        Args:
            device: 実行デバイス. "cuda" の場合は CUDA イベントを使用.
            skip_first: 最初の計測をスキップするか. ウォームアップ除外用.
        """
        self._use_cuda = device == "cuda" and torch.cuda.is_available()
        self._skip_first = skip_first
        self._total_time_ms: float = 0.0
        self._count: int = 0
        self._call_count: int = 0
        self._last_time_ms: float = 0.0

        # CUDA イベント (GPU 使用時のみ)
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None

        # CPU 用タイマー
        self._start_time: float = 0.0

    def start(self) -> None:
        """計測開始."""
        if self._use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()

    def stop(self) -> float:
        """計測終了.

        Returns:
            経過時間 (ミリ秒).
        """
        if self._use_cuda:
            if self._start_event is None or self._end_event is None:
                msg = "start() must be called before stop()"
                raise RuntimeError(msg)
            self._end_event.record()
            torch.cuda.synchronize()
            elapsed_ms: float = self._start_event.elapsed_time(self._end_event)
        else:
            elapsed_ms = (time.perf_counter() - self._start_time) * 1000

        self._call_count += 1
        self._last_time_ms = elapsed_ms

        # 最初の計測はスキップ (ウォームアップ)
        if self._skip_first and self._call_count == 1:
            return elapsed_ms

        self._total_time_ms += elapsed_ms
        self._count += 1

        return elapsed_ms

    @property
    def total_time_ms(self) -> float:
        """合計時間 (ミリ秒)."""
        return self._total_time_ms

    @property
    def count(self) -> int:
        """計測回数."""
        return self._count

    @property
    def last_time_ms(self) -> float:
        """直前の計測時間 (ミリ秒)."""
        return self._last_time_ms

    @property
    def average_time_ms(self) -> float:
        """平均時間 (ミリ秒).

        Returns:
            平均時間. 計測回数が 0 の場合は 0.0.
        """
        if self._count == 0:
            return 0.0
        return self._total_time_ms / self._count
