"""test_onnxパッケージ共通フィクスチャ."""

from pathlib import Path

import pytest

pytest.importorskip("onnx")

from pochidetection.models import RTDetrModel
from pochidetection.onnx import OnnxExporter

INPUT_SIZE = (64, 64)


@pytest.fixture(scope="session")
def onnx_path(
    rtdetr_model: RTDetrModel, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """エクスポート済みONNXファイルを作成するfixture."""
    tmp_dir = tmp_path_factory.mktemp("onnx")
    output_path = tmp_dir / "model.onnx"
    rtdetr_model.eval()
    exporter = OnnxExporter(model=rtdetr_model)
    exporter.export(output_path, input_size=INPUT_SIZE)
    return Path(output_path)
