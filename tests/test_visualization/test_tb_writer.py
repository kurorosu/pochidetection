"""TensorBoardWriter のユニットテスト."""

import logging
from pathlib import Path

from pochidetection.visualization.tensorboard import TensorBoardWriter


class TestTensorBoardWriter:
    """TensorBoardWriter のテスト."""

    def test_record_epoch_creates_event_file(self, tmp_path: "Path") -> None:
        """record_epoch で TensorBoard イベントファイルが作成されること."""
        logger = logging.getLogger("test_tb")
        writer = TensorBoardWriter(log_dir=tmp_path, logger=logger)

        writer.record_epoch(
            epoch=1,
            train_loss=0.5,
            val_loss=0.3,
            map_value=0.4,
            map_50=0.6,
            map_75=0.35,
            lr=0.001,
        )
        writer.close()

        event_files = list(tmp_path.glob("events.out.tfevents.*"))
        assert len(event_files) == 1

    def test_multiple_epochs(self, tmp_path: "Path") -> None:
        """複数エポックの記録が正常に動作すること."""
        logger = logging.getLogger("test_tb")
        writer = TensorBoardWriter(log_dir=tmp_path, logger=logger)

        for epoch in range(1, 4):
            writer.record_epoch(
                epoch=epoch,
                train_loss=0.5 / epoch,
                val_loss=0.4 / epoch,
                map_value=0.1 * epoch,
                map_50=0.15 * epoch,
                map_75=0.08 * epoch,
                lr=0.001 * (0.9**epoch),
            )
        writer.close()

        event_files = list(tmp_path.glob("events.out.tfevents.*"))
        assert len(event_files) == 1

    def test_close_is_idempotent(self, tmp_path: "Path") -> None:
        """close を複数回呼んでもエラーにならないこと."""
        logger = logging.getLogger("test_tb")
        writer = TensorBoardWriter(log_dir=tmp_path, logger=logger)

        writer.record_epoch(
            epoch=1,
            train_loss=0.5,
            val_loss=0.3,
            map_value=0.4,
            map_50=0.6,
            map_75=0.35,
            lr=0.001,
        )
        writer.close()
        writer.close()
