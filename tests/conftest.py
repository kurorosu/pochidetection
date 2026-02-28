"""テスト共通フィクスチャ."""

import pytest

from pochidetection.models import RTDetrModel
from pochidetection.utils import TrainingHistory


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
