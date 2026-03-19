"""物体検出モデルのインターフェース."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn as nn


class ModelOutputDict(TypedDict, total=False):
    """IDetectionModel.forward() の戻り値の基底型.

    学習時は ``loss`` を含む. 推論時の出力はアーキテクチャ固有の
    サブ型 (TransformerModelOutputDict, SSDModelOutputDict) を使用する.
    """

    loss: torch.Tensor


class TransformerModelOutputDict(ModelOutputDict, total=False):
    """Transformer ベースモデル (RT-DETR 等) の出力型.

    推論時は ``pred_logits`` と ``pred_boxes`` を含む.
    """

    pred_logits: torch.Tensor
    pred_boxes: torch.Tensor


class SSDModelOutputDict(ModelOutputDict, total=False):
    """SSD ベースモデル (SSD300, SSDLite 等) の出力型.

    推論時は ``predictions`` を含む.
    各要素は boxes (M, 4), scores (M,), labels (M,) を持つ辞書.
    """

    predictions: list[dict[str, torch.Tensor]]


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
    ) -> ModelOutputDict:
        """順伝播.

        Args:
            pixel_values: 入力画像テンソル, 形状は (B, C, H, W).
            labels: 学習時のターゲット. 各要素は以下のキーを含む辞書:
                - boxes: バウンディングボックス (N, 4)
                - class_labels: クラスラベル (N,)

        Returns:
            以下のキーを含む辞書:
            - loss: 学習時の損失 (labels が指定された場合)
            - モデル固有のキー:
                - RT-DETR: pred_logits (B, num_queries, num_classes),
                  pred_boxes (B, num_queries, 4)
                - SSDLite: predictions (list[dict], 各要素は
                  boxes (M, 4), scores (M,), labels (M,) を含む, 0-indexed)
        """
        pass

    @abstractmethod
    def save(self, save_dir: str | Path) -> None:
        """モデルを指定ディレクトリに保存.

        各実装はアーキテクチャ固有の永続化戦略をカプセル化する.

        Args:
            save_dir: 保存先ディレクトリパス.
        """
        pass

    @abstractmethod
    def load(self, load_dir: str | Path) -> None:
        """指定ディレクトリからモデルを復元.

        Args:
            load_dir: 読み込み元ディレクトリパス.
        """
        pass
