"""build_scheduler のテスト."""

import pytest
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)

from pochidetection.utils.scheduler import build_scheduler


@pytest.fixture()
def optimizer() -> torch.optim.Optimizer:
    """テスト用オプティマイザ."""
    model = torch.nn.Linear(10, 2)
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


class TestBuildScheduler:
    """build_scheduler のテスト."""

    def test_none_returns_none(self, optimizer: torch.optim.Optimizer) -> None:
        """scheduler_name=None で None を返す."""
        result = build_scheduler(optimizer, None, None, epochs=100)
        assert result is None

    def test_cosine_annealing_lr(self, optimizer: torch.optim.Optimizer) -> None:
        """CosineAnnealingLR を生成できる."""
        scheduler = build_scheduler(optimizer, "CosineAnnealingLR", None, epochs=50)
        assert isinstance(scheduler, CosineAnnealingLR)
        assert scheduler.T_max == 50

    def test_cosine_annealing_lr_with_custom_t_max(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        """CosineAnnealingLR で T_max を明示指定できる."""
        scheduler = build_scheduler(
            optimizer, "CosineAnnealingLR", {"T_max": 30}, epochs=50
        )
        assert isinstance(scheduler, CosineAnnealingLR)
        assert scheduler.T_max == 30

    def test_cosine_annealing_lr_with_eta_min(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        """CosineAnnealingLR で eta_min を指定できる."""
        scheduler = build_scheduler(
            optimizer,
            "CosineAnnealingLR",
            {"eta_min": 1e-6},
            epochs=100,
        )
        assert isinstance(scheduler, CosineAnnealingLR)
        assert scheduler.eta_min == 1e-6

    def test_step_lr(self, optimizer: torch.optim.Optimizer) -> None:
        """StepLR を生成できる."""
        scheduler = build_scheduler(optimizer, "StepLR", {"step_size": 10}, epochs=100)
        assert isinstance(scheduler, StepLR)

    def test_reduce_lr_on_plateau(self, optimizer: torch.optim.Optimizer) -> None:
        """ReduceLROnPlateau を生成できる."""
        scheduler = build_scheduler(
            optimizer, "ReduceLROnPlateau", {"patience": 5}, epochs=100
        )
        assert isinstance(scheduler, ReduceLROnPlateau)

    def test_invalid_scheduler_name(self, optimizer: torch.optim.Optimizer) -> None:
        """存在しない Scheduler 名で ValueError."""
        with pytest.raises(ValueError, match="不明な lr_scheduler"):
            build_scheduler(optimizer, "NonExistentScheduler", None, epochs=100)

    def test_non_scheduler_class(self, optimizer: torch.optim.Optimizer) -> None:
        """LRScheduler でないアトリビュート名で ValueError."""
        with pytest.raises(ValueError, match="LRScheduler のサブクラスではありません"):
            build_scheduler(optimizer, "EPOCH_DEPRECATION_WARNING", None, epochs=100)

    def test_lr_changes_with_cosine(self, optimizer: torch.optim.Optimizer) -> None:
        """CosineAnnealingLR で学習率が変化する."""
        scheduler = build_scheduler(optimizer, "CosineAnnealingLR", None, epochs=10)
        assert scheduler is not None

        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(5):
            scheduler.step()
        mid_lr = optimizer.param_groups[0]["lr"]

        assert mid_lr < initial_lr

    def test_scheduler_params_not_mutated(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        """渡した params dict が変更されないことを確認."""
        params = {"eta_min": 1e-6}
        original = params.copy()
        build_scheduler(optimizer, "CosineAnnealingLR", params, epochs=100)
        assert params == original
