"""フレーム出力先の抽象インターフェース."""

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

import numpy as np


class IFrameSink(ABC):
    """フレーム出力先の抽象基底クラス.

    動画ファイル書き出し・ディスプレイ表示等のフレームシンクを
    共通インターフェースで扱うための基底クラス.

    コンテキストマネージャプロトコルをサポートし,
    ``with`` 文で安全にリソースを解放できる.
    """

    @abstractmethod
    def write(self, frame: np.ndarray) -> None:
        """フレームを書き出す.

        Args:
            frame: BGR 形式の画像フレーム.
        """

    @abstractmethod
    def release(self) -> None:
        """リソースを解放する.

        ファイルハンドル・ウィンドウ等の外部リソースを閉じる.
        コンテキスト終了時や処理完了後に呼び出すこと.
        """

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
