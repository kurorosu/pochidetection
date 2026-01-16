"""DetectionLossのテスト."""

import pytest
import torch

from pochidetection.interfaces import IDetectionLoss
from pochidetection.losses import DetectionLoss


class TestDetectionLoss:
    """DetectionLossのテスト."""

    @pytest.fixture
    def loss_fn(self) -> DetectionLoss:
        """テスト用の損失関数を作成するfixture."""
        return DetectionLoss()

    def test_implements_interface(self, loss_fn: DetectionLoss) -> None:
        """IDetectionLossを実装していることを確認."""
        assert isinstance(loss_fn, IDetectionLoss)

    def test_call_returns_loss(self, loss_fn: DetectionLoss) -> None:
        """__call__が損失を返すことを確認."""
        outputs = {
            "loss": torch.tensor(1.5),
            "pred_boxes": torch.randn(2, 100, 4),
            "pred_logits": torch.randn(2, 100, 10),
        }
        targets: list[dict[str, torch.Tensor]] = []

        result = loss_fn(outputs, targets)

        assert isinstance(result, torch.Tensor)
        assert result.item() == 1.5

    def test_call_raises_error_when_loss_not_in_outputs(
        self, loss_fn: DetectionLoss
    ) -> None:
        """損失がoutputsに含まれていない場合にエラーが発生することを確認."""
        outputs = {
            "pred_boxes": torch.randn(2, 100, 4),
            "pred_logits": torch.randn(2, 100, 10),
        }
        targets: list[dict[str, torch.Tensor]] = []

        with pytest.raises(KeyError) as exc_info:
            loss_fn(outputs, targets)

        assert "loss" in str(exc_info.value)

    def test_custom_loss_key(self) -> None:
        """カスタム損失キーを指定できることを確認."""
        loss_fn = DetectionLoss(loss_key="total_loss")
        outputs = {
            "total_loss": torch.tensor(2.5),
            "pred_boxes": torch.randn(2, 100, 4),
            "pred_logits": torch.randn(2, 100, 10),
        }
        targets: list[dict[str, torch.Tensor]] = []

        result = loss_fn(outputs, targets)

        assert result.item() == 2.5
