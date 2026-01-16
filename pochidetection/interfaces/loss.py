"""物体検出損失関数のインターフェース."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class IDetectionLoss(ABC):
    """物体検出損失関数のインターフェース.

    すべての損失関数はこのインターフェースを実装し,
    トレーナーとの互換性を確保する.

    Note:
        RT-DETRなどTransformersベースのモデルはモデル内部で損失を計算するため,
        このインターフェースはカスタム損失関数を使用する場合に利用する.
    """

    @abstractmethod
    def __call__(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """損失を計算.

        Args:
            outputs: モデルの出力. 以下のキーを含む:
                - pred_boxes: 予測ボックス (B, num_queries, 4)
                - pred_logits: 予測ロジット (B, num_queries, num_classes)
            targets: ターゲットのリスト. 各要素は以下のキーを含む:
                - boxes: 正解ボックス (N, 4)
                - class_labels: クラスラベル (N,)

        Returns:
            スカラー損失テンソル.
        """
        pass
