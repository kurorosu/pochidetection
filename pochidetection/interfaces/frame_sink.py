"""フレーム出力先の抽象インターフェース."""

from abc import ABC, abstractmethod

import numpy as np


class IFrameSink(ABC):
    """フレーム出力先の抽象基底クラス.

    動画ファイル書き出し・ディスプレイ表示等のフレームシンクを
    共通インターフェースで扱うための基底クラス.
    """

    @abstractmethod
    def write(self, frame: np.ndarray) -> None:
        """フレームを書き出す.

        Args:
            frame: BGR 形式の画像フレーム.
        """

    @abstractmethod
    def release(self) -> None:
        """リソースを解放する."""
