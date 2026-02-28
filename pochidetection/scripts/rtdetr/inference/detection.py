"""検出結果のデータクラス."""

from dataclasses import dataclass


@dataclass
class Detection:
    """単一の検出結果.

    Attributes:
        box: バウンディングボックス [x1, y1, x2, y2] (ピクセル座標).
        score: 検出信頼度 (0.0 - 1.0).
        label: クラスラベル (整数).
    """

    box: list[float]
    score: float
    label: int
