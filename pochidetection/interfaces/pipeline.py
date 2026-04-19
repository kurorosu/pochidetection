"""推論パイプラインの抽象インターフェース."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import Generic, Literal, TypeVar

import numpy as np
import torch
from PIL import Image

from pochidetection.core.detection import Detection
from pochidetection.utils import PhasedTimer

type ImageInput = Image.Image | np.ndarray

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

    def __init__(self) -> None:
        """共通状態を初期化する.

        サブクラスは独自の ``__init__`` 冒頭で ``super().__init__()`` を呼び,
        ``self._pipeline_mode`` を Literal["cpu", "gpu"] で上書きする責務を負う.
        本基底で ``None`` を明示設定することで, ``pipeline_mode`` property の
        ``getattr`` fallback を撤廃し, 未初期化のサブクラスを早期に検出可能にする.
        """
        self._pipeline_mode: Literal["cpu", "gpu"] | None = None

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

    def _init_cuda_events(self, device: str) -> None:
        """CUDA Event を 1 度だけ生成してインスタンスにキャッシュする.

        Why: 推論毎に ``torch.cuda.Event(enable_timing=True)`` を新規生成すると
        アロケーション / GC コストが積み重なるため, ``__init__`` で 1 回だけ
        生成し ``record()`` で再利用する. 同一 Event への複数回 ``record()`` は
        PyTorch で許容されている.

        CUDA 利用不可 (CPU デバイス, CUDA 未利用環境) の場合は ``None`` を保持し,
        ``_measure_inference_gpu`` は ``time.perf_counter()`` 側の CPU 経路に退避する.

        Args:
            device: 実行デバイス文字列 (``"cuda"`` / ``"cuda:0"`` / ``"cpu"`` 等).
        """
        use_cuda = (
            isinstance(device, str) and "cuda" in device and torch.cuda.is_available()
        )
        if use_cuda:
            self._cuda_event_start: torch.cuda.Event | None = torch.cuda.Event(
                enable_timing=True
            )
            self._cuda_event_end: torch.cuda.Event | None = torch.cuda.Event(
                enable_timing=True
            )
        else:
            self._cuda_event_start = None
            self._cuda_event_end = None

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

        CUDA Event は ``__init__`` で生成済みのインスタンス変数
        ``self._cuda_event_start`` / ``self._cuda_event_end`` を再利用する.
        同一 Event への複数回 ``record()`` は PyTorch で許容されており,
        毎回の ``torch.cuda.Event()`` 新規生成によるアロケーション / GC コストを回避する.

        Yields:
            None.
        """
        start = getattr(self, "_cuda_event_start", None)
        end = getattr(self, "_cuda_event_end", None)
        if start is None or end is None:
            self._last_inference_gpu_ms = None
            yield
            return

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
    def pipeline_mode(self) -> Literal["cpu", "gpu"] | None:
        """Resolve 後の preprocess 経路 ('cpu' or 'gpu').

        Subclass の __init__ で ``self._pipeline_mode`` に保存された値を返す.
        基底 ``__init__`` が初期値として ``None`` を設定するため,
        ``super().__init__()`` を呼ばないサブクラスは ``AttributeError`` を送出し,
        未初期化を静かに 'cpu' に倒す挙動 (旧 ``getattr`` fallback) を防ぐ.
        Resolve は ``resolve_pipeline_mode()`` で行い, ONNX backend は常に 'cpu'.
        """
        return self._pipeline_mode
