"""物体検出用損失関数.

RT-DETRなどTransformersベースのモデルはモデル内部で損失を計算するため,
このモジュールはカスタム損失関数を使用する場合やモデル出力から損失を取得する場合に利用する.
"""

from typing import Any

import torch

from pochidetection.interfaces.loss import IDetectionLoss


class DetectionLoss(IDetectionLoss):
    """物体検出用損失関数.

    RT-DETRモデルが計算した損失をそのまま返すラッパー.
    モデルのforward出力に含まれる損失を取得する.

    Attributes:
        _loss_key: モデル出力から損失を取得するキー.
    """

    def __init__(self, loss_key: str = "loss") -> None:
        """DetectionLossを初期化.

        Args:
            loss_key: モデル出力辞書から損失を取得するキー.
        """
        self._loss_key = loss_key

    def __call__(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """損失を計算.

        モデル出力に含まれる損失をそのまま返す.
        RT-DETRなどのモデルはforward時にlabelsを渡すことで内部で損失を計算する.

        Args:
            outputs: モデルの出力. 以下のキーを含む:
                - loss: 損失テンソル (labelsを渡した場合)
                - pred_boxes: 予測ボックス (B, num_queries, 4)
                - pred_logits: 予測ロジット (B, num_queries, num_classes)
            targets: ターゲットのリスト (この実装では使用しない).

        Returns:
            スカラー損失テンソル.

        Raises:
            KeyError: モデル出力に損失が含まれていない場合.
        """
        if self._loss_key not in outputs:
            raise KeyError(
                f"モデル出力に '{self._loss_key}' が含まれていません. "
                f"forward時にlabelsを渡していることを確認してください."
            )
        return outputs[self._loss_key]
