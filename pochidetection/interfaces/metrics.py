"""物体検出評価指標のインターフェース."""

from abc import ABC, abstractmethod

import torch


class IDetectionMetrics(ABC):
    """物体検出評価指標のインターフェース.

    バッチ間で評価指標を計算するための統一インターフェースを提供.
    mAP (mean Average Precision) などの指標を計算する.
    """

    @abstractmethod
    def update(
        self,
        pred_boxes: list[torch.Tensor],
        pred_scores: list[torch.Tensor],
        pred_labels: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
        target_labels: list[torch.Tensor],
    ) -> None:
        """バッチ結果を蓄積.

        Args:
            pred_boxes: 予測ボックスのリスト. 各要素は (N, 4) の形状.
            pred_scores: 予測スコアのリスト. 各要素は (N,) の形状.
            pred_labels: 予測ラベルのリスト. 各要素は (N,) の形状.
            target_boxes: 正解ボックスのリスト. 各要素は (M, 4) の形状.
            target_labels: 正解ラベルのリスト. 各要素は (M,) の形状.
        """
        pass

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """蓄積した結果から指標を計算.

        Returns:
            指標名と値の辞書. 例: {"mAP": 0.75, "mAP_50": 0.85, "mAP_75": 0.70}
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """蓄積した状態をリセット."""
        pass
