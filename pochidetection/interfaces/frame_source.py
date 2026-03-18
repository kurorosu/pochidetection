"""フレーム供給元の抽象インターフェース."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType
from typing import Self

import numpy as np


class IFrameSource(ABC):
    """フレーム供給元の抽象基底クラス.

    動画ファイル・Webcam・RTSP ストリーム等のフレームソースを
    共通インターフェースで扱うための基底クラス.

    コンテキストマネージャプロトコルをサポートし,
    ``with`` 文で安全にリソースを解放できる.
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

    def __enter__(self) -> Self:
        """コンテキストマネージャの開始."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """コンテキストマネージャの終了. リソースを解放する."""
        self.release()
