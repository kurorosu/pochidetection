"""ONNX テスト用フィクスチャ."""

from pathlib import Path

import pytest

pytest.importorskip("onnx")

from pochidetection.models import RTDetrModel, SSDLiteModel
from pochidetection.onnx import RTDetrOnnxExporter, SSDLiteOnnxExporter

RTDETR_INPUT_SIZE = (64, 64)
SSDLITE_INPUT_SIZE = (64, 64)
SSDLITE_NUM_CLASSES = 2


@pytest.fixture(scope="session")
def rtdetr_onnx_path(
    rtdetr_model: RTDetrModel, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """エクスポート済みONNXファイルを作成するfixture."""
    tmp_dir = tmp_path_factory.mktemp("onnx")
    output_path = tmp_dir / "model.onnx"
    exporter = RTDetrOnnxExporter(model=rtdetr_model)
    exporter.export(output_path, input_size=RTDETR_INPUT_SIZE)
    return Path(output_path)


@pytest.fixture(scope="session")
def ssdlite_onnx_path(
    ssdlite_model: SSDLiteModel, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """FP32 エクスポート済み SSDLite ONNX ファイル."""
    tmp_dir = tmp_path_factory.mktemp("ssdlite_onnx")
    output_path = tmp_dir / "model.onnx"
    exporter = SSDLiteOnnxExporter(model=ssdlite_model)
    exporter.export(output_path, input_size=SSDLITE_INPUT_SIZE)
    return Path(output_path)


@pytest.fixture(scope="session")
def ssdlite_onnx_fp16_path(
    ssdlite_model: SSDLiteModel, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """FP16 エクスポート済み SSDLite ONNX ファイル."""
    tmp_dir = tmp_path_factory.mktemp("ssdlite_onnx_fp16")
    output_path = tmp_dir / "model_fp16.onnx"
    exporter = SSDLiteOnnxExporter(model=ssdlite_model)
    exporter.export(output_path, input_size=SSDLITE_INPUT_SIZE, fp16=True)
    return Path(output_path)
