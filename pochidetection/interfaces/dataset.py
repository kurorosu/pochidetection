"""物体検出データセットのインターフェース."""

from abc import ABC, abstractmethod
from typing import TypedDict

import torch


class DatasetSampleDict(TypedDict):
    """IDetectionDataset.__getitem__() の戻り値型.

    データセットが返す 1 サンプルの構造を定義する.
    ``pixel_values`` は前処理済み画像テンソル,
    ``labels`` はバウンディングボックスとクラスラベルを含む辞書.
    """

    pixel_values: torch.Tensor
    labels: dict[str, torch.Tensor]


class IDetectionDataset(ABC):
    """物体検出データセットのインターフェース.

    すべての物体検出データセットはこのインターフェースを実装する.
    """

    @abstractmethod
    def __len__(self) -> int:
        """データセット内のサンプル数を返す.

        Returns:
            サンプル数.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> DatasetSampleDict:
        """インデックスでサンプルを取得.

        Args:
            idx: サンプルのインデックス.

        Returns:
            以下のキーを含む辞書:
            - pixel_values: 前処理済み画像テンソル (C, H, W)
            - labels: ターゲット辞書 (boxes, class_labels を含む)
        """
        pass
