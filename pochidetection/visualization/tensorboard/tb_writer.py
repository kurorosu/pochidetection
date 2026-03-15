"""TensorBoard メトリクス記録."""

import logging
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TensorBoardWriter:
    """TensorBoard へのメトリクス記録を管理するクラス.

    学習中の loss, mAP, 学習率を TensorBoard に記録する.

    Args:
        log_dir: TensorBoard ログ出力ディレクトリ.
        logger: ロガーインスタンス.
    """

    def __init__(self, log_dir: Path, logger: logging.Logger) -> None:
        """初期化."""
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._logger = logger
        self._logger.debug(f"TensorBoard ログディレクトリ: {log_dir}")

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        map_value: float,
        map_50: float,
        map_75: float,
        lr: float,
    ) -> None:
        """1エポック分のメトリクスを TensorBoard に記録.

        Args:
            epoch: エポック番号.
            train_loss: 学習損失.
            val_loss: 検証損失.
            map_value: Mean Average Precision.
            map_50: mAP at IoU=0.50.
            map_75: mAP at IoU=0.75.
            lr: 現在の学習率.
        """
        self._writer.add_scalar("Loss/train", train_loss, epoch)
        self._writer.add_scalar("Loss/val", val_loss, epoch)
        self._writer.add_scalar("Metrics/mAP", map_value, epoch)
        self._writer.add_scalar("Metrics/mAP_50", map_50, epoch)
        self._writer.add_scalar("Metrics/mAP_75", map_75, epoch)
        self._writer.add_scalar("LearningRate", lr, epoch)

    def close(self) -> None:
        """ライターをフラッシュして閉じる."""
        self._writer.flush()
        self._writer.close()
        self._logger.debug("TensorBoard ライターを閉じました")
