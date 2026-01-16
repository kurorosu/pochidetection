"""物体検出モデルのインターフェース."""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class IDetectionModel(ABC, nn.Module):
    """物体検出モデルのインターフェース.

    すべての物体検出モデルはこのインターフェースを実装し,
    トレーナーや他のコンポーネントとの互換性を確保する.
    """

    @abstractmethod
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        """順伝播.

        Args:
            pixel_values: 入力画像テンソル, 形状は (B, C, H, W).
            labels: 学習時のターゲット. 各要素は以下のキーを含む辞書:
                - boxes: バウンディングボックス (N, 4)
                - class_labels: クラスラベル (N,)

        Returns:
            以下のキーを含む辞書:
            - loss: 学習時の損失 (labelsが指定された場合)
            - pred_boxes: 予測ボックス
            - pred_logits: 予測ロジット
        """
        pass

    @abstractmethod
    def get_backbone_params(self) -> list[nn.Parameter]:
        """層別学習率用のバックボーンパラメータを取得.

        Returns:
            バックボーンパラメータのリスト.
        """
        pass

    @abstractmethod
    def get_head_params(self) -> list[nn.Parameter]:
        """層別学習率用のヘッドパラメータを取得.

        Returns:
            ヘッドパラメータのリスト.
        """
        pass
