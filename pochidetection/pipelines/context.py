"""推論 pipeline の共通コンテキスト型とその構築ロジック.

``PipelineContext`` は推論 pipeline + PhasedTimer + Visualizer + InferenceSaver +
LabelMapper などを束ねる NamedTuple. ``ResolvedPipeline`` はモデル解決まで済んだ
後の全体結果 (context + config + model_path) を束ねる. ``InferenceContext`` は
``_run_inference`` / ``write_reports`` が受け取る最小面の Protocol.
``build_pipeline_context`` は ``setup_pipeline`` 内部から呼ばれ, LabelMapper /
Visualizer / InferenceSaver の構築を共通化する.
"""

from pathlib import Path
from typing import NamedTuple, Protocol

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.pipelines.model_path import (
    PRETRAINED,
    is_onnx_model,
    is_tensorrt_model,
)
from pochidetection.reporting import InferenceSaver, Visualizer
from pochidetection.utils import PhasedTimer
from pochidetection.visualization import LabelMapper

__all__ = [
    "InferenceContext",
    "PipelineContext",
    "ResolvedPipeline",
    "build_pipeline_context",
]


class PipelineContext(NamedTuple):
    """推論パイプラインコンテキスト."""

    pipeline: IDetectionPipeline
    phased_timer: PhasedTimer
    visualizer: Visualizer
    saver: InferenceSaver
    label_mapper: LabelMapper | None
    class_names: list[str] | None
    actual_device: str
    precision: str


class ResolvedPipeline(NamedTuple):
    """モデル解決・パイプライン構築の結果."""

    ctx: PipelineContext
    config: DetectionConfigDict
    config_path: str | None
    model_path: Path


class InferenceContext(Protocol):
    """推論コンテキストのプロトコル."""

    @property
    def pipeline(self) -> IDetectionPipeline:
        """推論パイプライン."""
        ...

    @property
    def phased_timer(self) -> PhasedTimer:
        """PhasedTimer."""
        ...

    @property
    def saver(self) -> InferenceSaver:
        """InferenceSaver."""
        ...

    @property
    def label_mapper(self) -> LabelMapper | None:
        """LabelMapper."""
        ...

    @property
    def class_names(self) -> list[str] | None:
        """クラス名リスト."""
        ...

    @property
    def actual_device(self) -> str:
        """実際のデバイス名."""
        ...

    @property
    def precision(self) -> str:
        """精度 (fp32/fp16)."""
        ...

    @property
    def visualizer(self) -> Visualizer:
        """Visualizer."""
        ...


def build_pipeline_context(
    *,
    pipeline: IDetectionPipeline,
    phased_timer: PhasedTimer,
    config: DetectionConfigDict,
    model_path: Path,
    actual_device: str,
    precision: str,
) -> PipelineContext:
    """共通の初期化ステップから PipelineContext を構築する.

    LabelMapper, Visualizer, InferenceSaver の構築を共通化する.

    Args:
        pipeline: 構築済みの推論パイプライン.
        phased_timer: 構築済みの PhasedTimer.
        config: 設定辞書.
        model_path: モデルのパス.
        actual_device: 実際のデバイス名.
        precision: 精度 (fp32/fp16).

    Returns:
        構築済みの PipelineContext.
    """
    class_names = config.get("class_names")
    label_mapper = LabelMapper(class_names) if class_names else None
    visualizer = Visualizer(label_mapper=label_mapper)

    if model_path == PRETRAINED:
        saver_base = Path(config.get("work_dir", "work_dirs")) / "pretrained"
    elif is_onnx_model(model_path) or is_tensorrt_model(model_path):
        saver_base = model_path.parent
    else:
        saver_base = model_path
    saver = InferenceSaver(saver_base)

    return PipelineContext(
        pipeline=pipeline,
        phased_timer=phased_timer,
        visualizer=visualizer,
        saver=saver,
        label_mapper=label_mapper,
        class_names=class_names,
        actual_device=actual_device,
        precision=precision,
    )
