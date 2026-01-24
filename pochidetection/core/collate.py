"""物体検出用Collate関数.

DataLoaderのバッチ作成時に使用するcollate関数を提供する.
"""

from typing import Any

import torch


class DetectionCollator:
    """物体検出用Collator.

    DataLoaderのcollate_fnとして使用し, バッチを作成する.
    CocoDetectionDatasetの出力形式に対応.

    入力形式 (各サンプル):
        - pixel_values: 前処理済み画像テンソル (C, H, W)
        - labels: {"boxes": tensor (N, 4), "class_labels": tensor (N,)}

    出力形式:
        - pixel_values: バッチ化された画像テンソル (B, C, H, W)
        - labels: ラベル辞書のリスト
    """

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """バッチをcollate.

        Args:
            batch: データセットからのサンプルリスト.
                各サンプルは以下のキーを含む:
                - pixel_values: 前処理済み画像テンソル (C, H, W)
                - labels: {"boxes": tensor (N, 4), "class_labels": tensor (N,)}

        Returns:
            以下のキーを含む辞書:
            - pixel_values: バッチ化された画像テンソル (B, C, H, W)
            - labels: ラベル辞書のリスト
        """
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = [item["labels"] for item in batch]
        return {"pixel_values": pixel_values, "labels": labels}
