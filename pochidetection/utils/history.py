"""学習履歴の管理クラス."""

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EpochMetrics:
    """1 epoch 分のメトリクス.

    Attributes:
        epoch: エポック番号 (1-indexed).
        train_loss: 学習損失.
        val_loss: 検証損失.
        mAP: Mean Average Precision.
        mAP_50: mAP at IoU=0.50.
        mAP_75: mAP at IoU=0.75.
        lr: 学習率.
    """

    epoch: int
    train_loss: float
    val_loss: float
    mAP: float
    mAP_50: float
    mAP_75: float
    lr: float


@dataclass
class TrainingHistory:
    """学習履歴の蓄積・保存.

    Attributes:
        records: エポックごとのメトリクスリスト.
    """

    records: list[EpochMetrics] = field(default_factory=list)

    def add(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        mAP: float,
        mAP_50: float,
        mAP_75: float,
        lr: float,
    ) -> None:
        """エポックのメトリクスを追加.

        Args:
            epoch: エポック番号 (1-indexed).
            train_loss: 学習損失.
            val_loss: 検証損失.
            mAP: Mean Average Precision.
            mAP_50: mAP at IoU=0.50.
            mAP_75: mAP at IoU=0.75.
            lr: 学習率.
        """
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            mAP=mAP,
            mAP_50=mAP_50,
            mAP_75=mAP_75,
            lr=lr,
        )
        self.records.append(metrics)

    def save_csv(self, path: Path) -> None:
        """CSV ファイルに保存.

        Args:
            path: 保存先パス.
        """
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "mAP",
            "mAP_50",
            "mAP_75",
            "lr",
        ]

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(
                    {
                        "epoch": record.epoch,
                        "train_loss": record.train_loss,
                        "val_loss": record.val_loss,
                        "mAP": record.mAP,
                        "mAP_50": record.mAP_50,
                        "mAP_75": record.mAP_75,
                        "lr": record.lr,
                    }
                )

    @classmethod
    def load_csv(cls, path: Path) -> "TrainingHistory":
        """CSV ファイルから読み込み.

        Args:
            path: 読み込み元パス.

        Returns:
            TrainingHistory インスタンス.
        """
        history = cls()

        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.add(
                    epoch=int(row["epoch"]),
                    train_loss=float(row["train_loss"]),
                    val_loss=float(row["val_loss"]),
                    mAP=float(row["mAP"]),
                    mAP_50=float(row["mAP_50"]),
                    mAP_75=float(row["mAP_75"]),
                    lr=float(row["lr"]),
                )

        return history

    @property
    def epochs(self) -> list[int]:
        """エポック番号のリストを取得."""
        return [r.epoch for r in self.records]

    @property
    def train_losses(self) -> list[float]:
        """学習損失のリストを取得."""
        return [r.train_loss for r in self.records]

    @property
    def val_losses(self) -> list[float]:
        """検証損失のリストを取得."""
        return [r.val_loss for r in self.records]

    @property
    def mAPs(self) -> list[float]:
        """Mean Average Precision のリストを取得."""
        return [r.mAP for r in self.records]

    @property
    def mAP_50s(self) -> list[float]:
        """mAP@50 のリストを取得."""
        return [r.mAP_50 for r in self.records]

    @property
    def mAP_75s(self) -> list[float]:
        """mAP@75 のリストを取得."""
        return [r.mAP_75 for r in self.records]
