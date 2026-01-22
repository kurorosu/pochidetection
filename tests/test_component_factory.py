"""ComponentFactoryのテスト."""

import pytest

from pochidetection.factories import ComponentFactory
from pochidetection.interfaces import (
    IDetectionDataset,
    IDetectionLoss,
    IDetectionMetrics,
    IDetectionModel,
)


class TestComponentFactory:
    """ComponentFactoryのテスト."""

    def test_get_available_models(self) -> None:
        """登録済みモデル名を取得できることを確認."""
        models = ComponentFactory.get_available_models()
        assert "RTDetr" in models

    def test_get_available_losses(self) -> None:
        """登録済み損失関数名を取得できることを確認."""
        losses = ComponentFactory.get_available_losses()
        assert "DetectionLoss" in losses

    def test_get_available_metrics(self) -> None:
        """登録済み評価指標名を取得できることを確認."""
        metrics = ComponentFactory.get_available_metrics()
        assert "DetectionMetrics" in metrics

    def test_get_available_datasets(self) -> None:
        """登録済みデータセット名を取得できることを確認."""
        datasets = ComponentFactory.get_available_datasets()
        assert "CocoDetectionDataset" in datasets

    def test_create_model(self) -> None:
        """設定からモデルを生成できることを確認."""
        config = {
            "architecture": "RTDetr",
            "model_name": "PekingU/rtdetr_r18vd",
            "num_classes": 2,
            "pretrained": False,
        }
        model = ComponentFactory.create_model(config)
        assert isinstance(model, IDetectionModel)

    def test_create_loss(self) -> None:
        """設定から損失関数を生成できることを確認."""
        config = {
            "loss": "DetectionLoss",
            "loss_params": {},
        }
        loss = ComponentFactory.create_loss(config)
        assert isinstance(loss, IDetectionLoss)

    def test_create_metrics(self) -> None:
        """設定から評価指標を生成できることを確認."""
        config = {
            "metrics": "DetectionMetrics",
            "metrics_params": {},
        }
        metrics = ComponentFactory.create_metrics(config)
        assert isinstance(metrics, IDetectionMetrics)

    def test_create_dataset(self, tmp_path: pytest.TempPathFactory) -> None:
        """設定からデータセットを生成できることを確認."""
        # テスト用のダミーアノテーションを作成
        import json

        image_dir = tmp_path / "images"  # type: ignore
        image_dir.mkdir()

        annotation_file = tmp_path / "annotations.json"  # type: ignore
        annotation_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "cat"}],
        }
        annotation_file.write_text(json.dumps(annotation_data))

        config = {
            "dataset": "CocoDetectionDataset",
            "image_dir": str(image_dir),
            "annotation_file": str(annotation_file),
        }
        dataset = ComponentFactory.create_dataset(config)
        assert isinstance(dataset, IDetectionDataset)

    def test_create_model_unknown_architecture(self) -> None:
        """未登録のモデル名でエラーが発生することを確認."""
        config = {
            "architecture": "UnknownModel",
            "num_classes": 2,
        }
        with pytest.raises(ValueError, match="未登録のモデル"):
            ComponentFactory.create_model(config)

    def test_create_loss_unknown_loss(self) -> None:
        """未登録の損失関数名でエラーが発生することを確認."""
        config = {
            "loss": "UnknownLoss",
        }
        with pytest.raises(ValueError, match="未登録の損失関数"):
            ComponentFactory.create_loss(config)

    def test_create_metrics_unknown_metrics(self) -> None:
        """未登録の評価指標名でエラーが発生することを確認."""
        config = {
            "metrics": "UnknownMetrics",
        }
        with pytest.raises(ValueError, match="未登録の評価指標"):
            ComponentFactory.create_metrics(config)

    def test_create_dataset_unknown_dataset(self) -> None:
        """未登録のデータセット名でエラーが発生することを確認."""
        config = {
            "dataset": "UnknownDataset",
            "image_dir": "/tmp/images",
        }
        with pytest.raises(ValueError, match="未登録のデータセット"):
            ComponentFactory.create_dataset(config)
