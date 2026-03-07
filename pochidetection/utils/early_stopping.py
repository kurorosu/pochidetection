"""Early Stopping コールバック."""

from typing import Literal


class EarlyStopping:
    """Early Stopping による学習の早期終了を判定するクラス.

    監視メトリクスが patience エポック連続で改善しない場合に停止を通知する.

    Args:
        patience: 改善なしで停止するまでのエポック数.
        min_delta: 改善と見なす最小変化量.
        mode: "min" (val_loss 等) or "max" (mAP 等).
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "max",
    ) -> None:
        """初期化."""
        if patience < 1:
            raise ValueError(f"patience は 1 以上である必要があります: {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta は 0 以上である必要があります: {min_delta}")

        self._patience = patience
        self._min_delta = min_delta
        self._mode = mode
        self._counter = 0
        self._best_value: float | None = None
        self._best_epoch = 0

    @property
    def patience(self) -> int:
        """改善なしで停止するまでのエポック数."""
        return self._patience

    @property
    def counter(self) -> int:
        """改善なしの連続エポック数."""
        return self._counter

    @property
    def best_value(self) -> float | None:
        """最良メトリクス値."""
        return self._best_value

    @property
    def best_epoch(self) -> int:
        """最良メトリクスを記録したエポック (1-indexed)."""
        return self._best_epoch

    def step(self, current_value: float, epoch: int) -> bool:
        """メトリクスを更新し, 停止すべきかを判定.

        Args:
            current_value: 現在のメトリクス値.
            epoch: 現在のエポック番号 (1-indexed).

        Returns:
            True なら学習を停止すべき.
        """
        if self._best_value is None or self._is_improvement(current_value):
            self._best_value = current_value
            self._best_epoch = epoch
            self._counter = 0
            return False

        self._counter += 1
        return self._counter >= self._patience

    def _is_improvement(self, current_value: float) -> bool:
        """現在の値が改善かどうかを判定."""
        assert self._best_value is not None
        if self._mode == "max":
            return current_value > self._best_value + self._min_delta
        return current_value < self._best_value - self._min_delta
