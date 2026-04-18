"""推論パイプラインの抽象インターフェース."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import Generic, Literal, TypeAlias, TypeVar

import numpy as np
import torch
from PIL import Image

from pochidetection.core.detection import Detection
from pochidetection.utils import PhasedTimer

ImageInput: TypeAlias = Image.Image | np.ndarray

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

    @contextmanager
    def _measure_inference_gpu(self) -> Generator[None]:
        """CUDA Event で inference 区間の GPU 実行時間を計測.

        Why: 既存 _measure は wall-clock のため GIL / asyncio 待ち時間を含む.
        本ヘルパーは GPU クロック由来の純粋な kernel 実行時間を別途記録し,
        wall-clock との差分から Python 側の待ち時間を切り分けられるようにする.

        計測結果は ``self._last_inference_gpu_ms`` に格納される.
        ``self._device`` が cuda かつ CUDA 利用可能な場合のみ計測.
        それ以外は None を保持し yield して素通りする.

        Yields:
            None.
        """
        device = getattr(self, "_device", "cpu")
        use_cuda = (
            isinstance(device, str) and "cuda" in device and torch.cuda.is_available()
        )
        if not use_cuda:
            self._last_inference_gpu_ms = None
            yield
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            torch.cuda.synchronize()
            self._last_inference_gpu_ms = start.elapsed_time(end)

    @property
    def last_inference_gpu_ms(self) -> float | None:
        """直近の inference 区間の GPU 実行時間 (CUDA Event 計測, ms).

        ``_measure_inference_gpu`` で記録された値. CUDA 不可または未計測時は None.
        """
        return getattr(self, "_last_inference_gpu_ms", None)

    @abstractmethod
    def run(self, image: ImageInput) -> list[Detection]:
        """E2E 推論を実行する.

        Args:
            image: 入力画像 (PIL Image または numpy RGB 配列).

        Returns:
            検出結果のリスト.
        """

    @property
    def phased_timer(self) -> PhasedTimer | None:
        """フェーズ別タイマーを取得."""
        return self._phased_timer

    @property
    def pipeline_mode(self) -> Literal["cpu", "gpu"]:
        """Resolve 後の preprocess 経路 ('cpu' or 'gpu').

        Subclass の __init__ で ``self._pipeline_mode`` に保存された値を返す.
        Resolve は ``resolve_pipeline_mode()`` で行い, ONNX backend は常に 'cpu'.
        """
        return getattr(self, "_pipeline_mode", "cpu")
