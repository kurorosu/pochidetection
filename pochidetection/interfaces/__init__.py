"""物体検出コンポーネントのインターフェース群を提供."""

from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.interfaces.dataset import DatasetSampleDict, IDetectionDataset
from pochidetection.interfaces.frame_sink import IFrameSink
from pochidetection.interfaces.frame_source import IFrameSource
from pochidetection.interfaces.model import IDetectionModel, ModelOutputDict
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.interfaces.plotter import IReportPlotter, ITrainingCurvePlotter

__all__ = [
    "DatasetSampleDict",
    "IDetectionDataset",
    "IDetectionModel",
    "IFrameSink",
    "IFrameSource",
    "ModelOutputDict",
    "IDetectionPipeline",
    "IInferenceBackend",
    "IReportPlotter",
    "ITrainingCurvePlotter",
]
