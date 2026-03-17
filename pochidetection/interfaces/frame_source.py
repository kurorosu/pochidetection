"""フレーム供給元の抽象インターフェース."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np


class IFrameSource(ABC):
    """フレーム供給元の抽象基底クラス.

    動画ファイル・Webcam・RTSP ストリーム等のフレームソースを
    共通インターフェースで扱うための基底クラス.
    """

    @property
    @abstractmethod
    def fps(self) -> float:
        """フレームレートを取得."""

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """フレームサイズを (width, height) で取得."""

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        """フレームを順次返すイテレータ."""

    @abstractmethod
    def release(self) -> None:
        """リソースを解放する."""
