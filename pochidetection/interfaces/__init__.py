"""物体検出コンポーネントのインターフェース群を提供."""

from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.interfaces.dataset import DatasetSampleDict, IDetectionDataset
from pochidetection.interfaces.model import IDetectionModel, ModelOutputDict
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.interfaces.plotter import IReportPlotter, ITrainingCurvePlotter

__all__ = [
    "DatasetSampleDict",
    "IDetectionDataset",
    "IDetectionModel",
    "ModelOutputDict",
    "IDetectionPipeline",
    "IInferenceBackend",
    "IReportPlotter",
    "ITrainingCurvePlotter",
]
