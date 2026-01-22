"""DetectionTrainerのテスト."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from pochidetection.interfaces import (
    IDetectionLoss,
    IDetectionMetrics,
    IDetectionModel,
)
from pochidetection.trainer import DetectionTrainer


class MockModel(IDetectionModel):
    """テスト用モックモデル."""

    def __init__(self) -> None:
        """初期化."""
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """順伝播."""
        batch_size = pixel_values.shape[0]
        result: dict[str, torch.Tensor] = {
            "pred_logits": torch.randn(batch_size, 100, 2),
            "pred_boxes": torch.rand(batch_size, 100, 4),
        }
        if labels is not None:
            result["loss"] = torch.tensor(0.5, requires_grad=True)
        return result

    def get_backbone_params(self) -> list[torch.nn.Parameter]:
        """バックボーンパラメータ."""
        return []

    def get_head_params(self) -> list[torch.nn.Parameter]:
        """ヘッドパラメータ."""
        return list(self.parameters())


class MockLoss(IDetectionLoss):
    """テスト用モック損失関数."""

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """損失計算."""
        return torch.tensor(0.5, requires_grad=True)


class MockMetrics(IDetectionMetrics):
    """テスト用モック評価指標."""

    def __init__(self) -> None:
        """初期化."""
        self._count = 0

    def update(
        self,
        pred_boxes: list[torch.Tensor],
        pred_scores: list[torch.Tensor],
        pred_labels: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
        target_labels: list[torch.Tensor],
    ) -> None:
        """更新."""
        self._count += 1

    def compute(self) -> dict[str, float]:
        """計算."""
        return {"mAP": 0.5, "mAP_50": 0.7, "mAP_75": 0.4}

    def reset(self) -> None:
        """リセット."""
        self._count = 0


class MockDataset(Dataset[dict[str, torch.Tensor]]):
    """テスト用モックデータセット."""

    def __init__(self, size: int = 10) -> None:
        """初期化."""
        self._size = size

    def __len__(self) -> int:
        """サイズ."""
        return self._size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """アイテム取得."""
        return {
            "pixel_values": torch.randn(3, 640, 640),
            "boxes": torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
            "labels": torch.tensor([0]),
            "image_id": idx,
            "orig_size": (640, 640),
        }


def mock_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """テスト用collate関数."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    return {
        "pixel_values": pixel_values,
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
        "orig_sizes": [b["orig_size"] for b in batch],
    }


class TestDetectionTrainer:
    """DetectionTrainerのテスト."""

    @pytest.fixture
    def trainer(self, tmp_path: pytest.TempPathFactory) -> DetectionTrainer:
        """Trainerフィクスチャ."""
        model = MockModel()
        loss_fn = MockLoss()
        metrics = MockMetrics()
        optimizer = SGD(model.parameters(), lr=0.01)

        return DetectionTrainer(
            model=model,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            device="cpu",
            work_dir=tmp_path,  # type: ignore
            use_amp=False,
        )

    @pytest.fixture
    def train_loader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """学習データローダーフィクスチャ."""
        dataset = MockDataset(size=4)
        return DataLoader(dataset, batch_size=2, collate_fn=mock_collate)

    @pytest.fixture
    def val_loader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """検証データローダーフィクスチャ."""
        dataset = MockDataset(size=4)
        return DataLoader(dataset, batch_size=2, collate_fn=mock_collate)

    def test_trainer_initialization(self, trainer: DetectionTrainer) -> None:
        """Trainerの初期化を確認."""
        assert trainer.best_map == 0.0
        assert trainer.history["train_loss"] == []
        assert trainer.history["val_loss"] == []

    def test_train_one_epoch(
        self,
        trainer: DetectionTrainer,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """1エポックの学習を確認."""
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
        )

        assert len(history["train_loss"]) == 1
        assert len(history["val_loss"]) == 1
        assert len(history["mAP"]) == 1

    def test_train_saves_best_model(
        self,
        trainer: DetectionTrainer,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """ベストモデルが保存されることを確認."""
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
        )

        best_path = tmp_path / "best.pth"  # type: ignore
        assert best_path.exists()

    def test_train_saves_last_model(
        self,
        trainer: DetectionTrainer,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """最終モデルが保存されることを確認."""
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
        )

        last_path = tmp_path / "last.pth"  # type: ignore
        assert last_path.exists()

    def test_load_checkpoint(
        self,
        trainer: DetectionTrainer,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """チェックポイントの読み込みを確認."""
        # 学習してチェックポイントを作成
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
        )

        # 新しいTrainerを作成
        new_model = MockModel()
        new_trainer = DetectionTrainer(
            model=new_model,
            loss_fn=MockLoss(),
            metrics=MockMetrics(),
            optimizer=SGD(new_model.parameters(), lr=0.01),
            device="cpu",
            work_dir=tmp_path,  # type: ignore
            use_amp=False,
        )

        # チェックポイントを読み込み
        epoch = new_trainer.load_checkpoint(tmp_path / "best.pth")  # type: ignore
        assert epoch == 1

    def test_history_property(
        self,
        trainer: DetectionTrainer,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """historyプロパティを確認."""
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
        )

        history = trainer.history
        assert "train_loss" in history
        assert "val_loss" in history
        assert "mAP" in history
        assert "mAP_50" in history
        assert "mAP_75" in history
        assert len(history["train_loss"]) == 2
