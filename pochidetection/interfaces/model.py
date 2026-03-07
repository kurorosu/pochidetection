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
            - loss: 学習時の損失 (labels が指定された場合, 必須)
            - predictions: 推論時の検出結果 (labels が None の場合, 必須).
                list[dict] で各要素は boxes (M, 4), scores (M,),
                labels (M,) を含む. labels は 0-indexed.
            上記に加え, モデル固有のキー (pred_logits, pred_boxes 等)
            を含んでもよい.
        """
        pass
