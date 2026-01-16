"""物体検出データセットのインターフェース."""

from abc import ABC, abstractmethod
from typing import Any


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
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """インデックスでサンプルを取得.

        Args:
            idx: サンプルのインデックス.

        Returns:
            以下のキーを含む辞書:
            - image: 画像テンソル (C, H, W)
            - boxes: バウンディングボックス (N, 4) [x_min, y_min, x_max, y_max]
            - labels: クラスラベル (N,)
            - image_id: 画像ID
        """
        pass

    @abstractmethod
    def get_categories(self) -> list[dict[str, Any]]:
        """カテゴリ情報を取得.

        Returns:
            カテゴリ情報のリスト. 各要素は {"id": int, "name": str} の形式.
        """
        pass
