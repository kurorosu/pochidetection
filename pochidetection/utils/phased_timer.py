"""フェーズ別の推論時間計測クラス."""

from collections.abc import Generator
from contextlib import contextmanager

from pochidetection.utils.timer import InferenceTimer


class PhasedTimer:
    """フェーズ別の推論時間計測.

    内部にフェーズ名をキーとした InferenceTimer の辞書を持ち,
    preprocess / inference / postprocess 等のフェーズ別に時間を計測する.

    Attributes:
        _timers: フェーズ名をキーとした InferenceTimer の辞書.
    """

    def __init__(
        self,
        phases: list[str],
        device: str = "cuda",
        skip_first: bool = True,
    ) -> None:
        """初期化.

        Args:
            phases: 計測対象のフェーズ名リスト.
                例: ["preprocess", "inference", "postprocess"]
            device: 実行デバイス. "cuda" の場合は CUDA イベントを使用.
            skip_first: 最初の計測をスキップするか. ウォームアップ除外用.

        Raises:
            ValueError: phases が空, または重複を含む場合.
        """
        if not phases:
            raise ValueError("phases must not be empty.")
        if len(phases) != len(set(phases)):
            raise ValueError("phases must not contain duplicates.")

        self._timers: dict[str, InferenceTimer] = {
            phase: InferenceTimer(device=device, skip_first=skip_first)
            for phase in phases
        }

    def _get_timer(self, phase: str) -> InferenceTimer:
        """指定フェーズのタイマーを取得 (内部用).

        Args:
            phase: フェーズ名.

        Returns:
            対応する InferenceTimer.

        Raises:
            ValueError: 未登録のフェーズ名が指定された場合.
        """
        if phase not in self._timers:
            raise ValueError(f"Unknown phase: '{phase}'. Available: {self.phases}")
        return self._timers[phase]

    @contextmanager
    def measure(self, phase: str) -> Generator[None, None, None]:
        """指定フェーズの計測をコンテキストマネージャで実行.

        Args:
            phase: 計測対象のフェーズ名.

        Yields:
            None.

        Raises:
            ValueError: 未登録のフェーズ名が指定された場合.
        """
        with self._get_timer(phase).measure():
            yield

    def get_timer(self, phase: str) -> InferenceTimer:
        """指定フェーズのタイマーを取得.

        Args:
            phase: フェーズ名.

        Returns:
            対応する InferenceTimer.

        Raises:
            ValueError: 未登録のフェーズ名が指定された場合.
        """
        return self._get_timer(phase)

    @property
    def phases(self) -> list[str]:
        """登録済みフェーズ名一覧."""
        return list(self._timers.keys())

    def summary(self) -> dict[str, dict[str, int | float]]:
        """全フェーズの計測結果を辞書で返す.

        Returns:
            フェーズ名をキーとした計測結果の辞書.
            各フェーズは total_ms (float), count (int), average_ms (float) を含む.
        """
        return {
            phase: {
                "total_ms": timer.total_time_ms,
                "count": timer.count,
                "average_ms": timer.average_time_ms,
            }
            for phase, timer in self._timers.items()
        }

    def reset(self) -> None:
        """全フェーズのタイマーをリセット."""
        for timer in self._timers.values():
            timer.reset()
