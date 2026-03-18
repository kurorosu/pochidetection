"""推論パイプラインの抽象インターフェース."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import Generic, TypeVar

from PIL import Image

from pochidetection.core.detection import Detection
from pochidetection.utils import PhasedTimer

TPreprocessed = TypeVar("TPreprocessed")
TInferred = TypeVar("TInferred")


class IDetectionPipeline(ABC, Generic[TPreprocessed, TInferred]):
    """推論パイプラインの共通インターフェース.

    前処理・推論・後処理の 3 フェーズで構成される E2E 推論パイプラインの
    共通契約を定義する. 型パラメータによりフェーズ間のデータ型を明示する.

    Type Parameters:
        TPreprocessed: preprocess の出力型.
        TInferred: infer の出力型.

    サブクラスは ``run`` メソッドで preprocess → infer → postprocess を
    順に実行し, ``_measure`` ヘルパーでフェーズ別計測を行う.

    phased_timer の検証・保持・計測ヘルパーを提供し,
    サブクラスでの重複を防止する.
    """

    PHASES = ["preprocess", "inference", "postprocess"]

    def _validate_phased_timer(self, phased_timer: PhasedTimer | None) -> None:
        """phased_timer のフェーズ構成を検証し, インスタンスに保持する.

        Args:
            phased_timer: フェーズ別タイマー. None の場合は計測しない.

        Raises:
            ValueError: phased_timer に必須フェーズが含まれていない場合.
        """
        if phased_timer is not None:
            missing = set(self.PHASES) - set(phased_timer.phases)
            if missing:
                raise ValueError(
                    f"phased_timer is missing required phases: {sorted(missing)}. "
                    f"Required: {self.PHASES}"
                )
        self._phased_timer = phased_timer

    @contextmanager
    def _measure(self, phase: str) -> Generator[None]:
        """フェーズ計測のコンテキストマネージャ.

        phased_timer が設定されている場合は計測し, None の場合は素通りする.

        Args:
            phase: 計測対象のフェーズ名.

        Yields:
            None.
        """
        if self._phased_timer is not None:
            with self._phased_timer.measure(phase):
                yield
        else:
            yield

    @abstractmethod
    def run(self, image: Image.Image) -> list[Detection]:
        """E2E 推論を実行する.

        Args:
            image: 入力画像 (PIL Image).

        Returns:
            検出結果のリスト.
        """

    @property
    def phased_timer(self) -> PhasedTimer | None:
        """フェーズ別タイマーを取得."""
        return self._phased_timer
