"""学習履歴の管理クラス."""

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _EpochMetrics:
    """1 epoch 分のメトリクス.

    Attributes:
        epoch: エポック番号 (1-indexed).
        train_loss: 学習損失.
        val_loss: 検証損失.
        map: Mean Average Precision.
        map_50: mAP at IoU=0.50.
        map_75: mAP at IoU=0.75.
        lr: 学習率.
    """

    epoch: int
    train_loss: float
    val_loss: float
    map: float
    map_50: float
    map_75: float
    lr: float


@dataclass
class TrainingHistory:
    """学習履歴の蓄積・保存.

    Attributes:
        records: エポックごとのメトリクスリスト.
    """

    records: list[_EpochMetrics] = field(default_factory=list)

    def add(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        map: float,
        map_50: float,
        map_75: float,
        lr: float,
    ) -> None:
        """エポックのメトリクスを追加.

        Args:
            epoch: エポック番号 (1-indexed).
            train_loss: 学習損失.
            val_loss: 検証損失.
            map: Mean Average Precision.
            map_50: mAP at IoU=0.50.
            map_75: mAP at IoU=0.75.
            lr: 学習率.
        """
        metrics = _EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            map=map,
            map_50=map_50,
            map_75=map_75,
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
            "map",
            "map_50",
            "map_75",
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
                        "map": record.map,
                        "map_50": record.map_50,
                        "map_75": record.map_75,
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
                    map=float(row["map"]),
                    map_50=float(row["map_50"]),
                    map_75=float(row["map_75"]),
                    lr=float(row["lr"]),
                )

        return history

    @property
    def epochs(self) -> list[int]:
        """エポック番号のリストを取得.

        Returns:
            エポック番号のリスト.
        """
        return [r.epoch for r in self.records]

    @property
    def train_losses(self) -> list[float]:
        """学習損失のリストを取得.

        Returns:
            学習損失のリスト.
        """
        return [r.train_loss for r in self.records]

    @property
    def val_losses(self) -> list[float]:
        """検証損失のリストを取得.

        Returns:
            検証損失のリスト.
        """
        return [r.val_loss for r in self.records]

    @property
    def maps(self) -> list[float]:
        """Mean Average Precision のリストを取得.

        Returns:
            mAP のリスト.
        """
        return [r.map for r in self.records]

    @property
    def map_50s(self) -> list[float]:
        """mAP@50 のリストを取得.

        Returns:
            mAP@50 のリスト.
        """
        return [r.map_50 for r in self.records]

    @property
    def map_75s(self) -> list[float]:
        """mAP@75 のリストを取得.

        Returns:
            mAP@75 のリスト.
        """
        return [r.map_75 for r in self.records]
