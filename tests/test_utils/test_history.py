"""TrainingHistory のテスト."""

from pathlib import Path

from pochidetection.utils import TrainingHistory


class TestTrainingHistory:
    """TrainingHistory クラスのテスト."""

    def test_add_records(self) -> None:
        """レコードを追加できることを確認."""
        history = TrainingHistory()

        history.add(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            map=0.3,
            map_50=0.5,
            map_75=0.2,
            lr=0.001,
        )

        assert len(history.records) == 1
        assert history.records[0].epoch == 1
        assert history.records[0].train_loss == 0.5

    def test_property_accessors(self) -> None:
        """プロパティでリストを取得できることを確認."""
        history = TrainingHistory()

        history.add(1, 0.5, 0.4, 0.3, 0.5, 0.2, 0.001)
        history.add(2, 0.4, 0.3, 0.4, 0.6, 0.3, 0.001)

        assert history.epochs == [1, 2]
        assert history.train_losses == [0.5, 0.4]
        assert history.val_losses == [0.4, 0.3]
        assert history.maps == [0.3, 0.4]
        assert history.map_50s == [0.5, 0.6]
        assert history.map_75s == [0.2, 0.3]

    def test_save_and_load_csv(self, tmp_path: Path) -> None:
        """CSV の保存と読み込みが正しく動作することを確認."""
        history = TrainingHistory()
        history.add(1, 0.5432, 0.4321, 0.1234, 0.2345, 0.0987, 0.001)
        history.add(2, 0.4321, 0.3456, 0.2345, 0.3456, 0.1234, 0.0005)

        csv_path = tmp_path / "history.csv"
        history.save_csv(csv_path)

        # ファイルが作成されたことを確認
        assert csv_path.exists()

        # 読み込み
        loaded = TrainingHistory.load_csv(csv_path)

        assert len(loaded.records) == 2
        assert loaded.records[0].epoch == 1
        assert loaded.records[0].train_loss == 0.5432
        assert loaded.records[1].epoch == 2
        assert loaded.records[1].lr == 0.0005

    def test_csv_format(self, tmp_path: Path) -> None:
        """CSV フォーマットが正しいことを確認."""
        history = TrainingHistory()
        history.add(1, 0.5, 0.4, 0.3, 0.5, 0.2, 0.001)

        csv_path = tmp_path / "history.csv"
        history.save_csv(csv_path)

        content = csv_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")

        # ヘッダー確認
        assert lines[0] == "epoch,train_loss,val_loss,map,map_50,map_75,lr"
        # データ行確認
        assert lines[1] == "1,0.5,0.4,0.3,0.5,0.2,0.001"
