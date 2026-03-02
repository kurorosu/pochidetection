"""テスト共通フィクスチャ."""

from pathlib import Path

import pytest

pytest.importorskip("onnx")

from pochidetection.models import RTDetrModel
from pochidetection.onnx import OnnxExporter
from pochidetection.utils import TrainingHistory

ONNX_INPUT_SIZE = (64, 64)


@pytest.fixture(scope="session")
def rtdetr_model() -> RTDetrModel:
    """テスト用の軽量RTDetrModelを作成するsessionスコープfixture.

    全テストで1つのインスタンスを共有し, モデル初期化コストを削減する.
    """
    model = RTDetrModel(
        model_name="PekingU/rtdetr_r18vd", num_classes=2, pretrained=False
    )
    model.model.config.num_queries = 50
    return model


@pytest.fixture(scope="session")
def onnx_path(
    rtdetr_model: RTDetrModel, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """エクスポート済みONNXファイルを作成するfixture."""
    tmp_dir = tmp_path_factory.mktemp("onnx")
    output_path = tmp_dir / "model.onnx"
    rtdetr_model.eval()
    exporter = OnnxExporter(model=rtdetr_model)
    exporter.export(output_path, input_size=ONNX_INPUT_SIZE)
    return Path(output_path)


@pytest.fixture(scope="class")
def training_history() -> TrainingHistory:
    """3エポック分の TrainingHistory を作成するfixture."""
    history = TrainingHistory()
    history.add(1, 0.5, 0.4, 0.3, 0.5, 0.2, 0.001)
    history.add(2, 0.4, 0.3, 0.4, 0.6, 0.3, 0.001)
    history.add(3, 0.3, 0.25, 0.5, 0.7, 0.4, 0.0005)
    return history


@pytest.fixture(scope="class")
def single_epoch_history() -> TrainingHistory:
    """1エポック分の TrainingHistory を作成するfixture."""
    history = TrainingHistory()
    history.add(1, 0.5, 0.4, 0.3, 0.5, 0.2, 0.001)
    return history
