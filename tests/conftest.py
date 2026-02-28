"""テスト共通フィクスチャ."""

import pytest

from pochidetection.utils import TrainingHistory


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
