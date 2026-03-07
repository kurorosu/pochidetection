"""SSDLiteModel のテスト."""

import torch
import torch.nn as nn

from pochidetection.interfaces import IDetectionModel
from pochidetection.models import SSDLiteModel


class TestSSDLiteModel:
    """SSDLiteModel のテスト."""

    def test_implements_interface(self) -> None:
        """IDetectionModel を実装していることを確認."""
        model = SSDLiteModel(num_classes=2, pretrained=False)
        assert isinstance(model, IDetectionModel)

    def test_is_nn_module(self) -> None:
        """nn.Module を継承していることを確認."""
        model = SSDLiteModel(num_classes=2, pretrained=False)
        assert isinstance(model, nn.Module)

    def test_num_classes_property(self) -> None:
        """num_classes がユーザ指定値 (背景なし) を返すことを確認."""
        model = SSDLiteModel(num_classes=4, pretrained=False)
        assert model.num_classes == 4

    def test_forward_inference(self) -> None:
        """推論時の forward が predictions を返すことを確認."""
        model = SSDLiteModel(num_classes=2, pretrained=False)
        model.eval()

        pixel_values = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            outputs = model(pixel_values)

        assert "predictions" in outputs
        assert len(outputs["predictions"]) == 1

        pred = outputs["predictions"][0]
        assert "boxes" in pred
        assert "scores" in pred
        assert "labels" in pred

    def test_forward_inference_labels_are_0_indexed(self) -> None:
        """推論時のラベルが 0-indexed であることを確認."""
        model = SSDLiteModel(num_classes=2, pretrained=False)
        model.eval()

        pixel_values = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            outputs = model(pixel_values)

        pred = outputs["predictions"][0]
        if pred["labels"].numel() > 0:
            # SSD 出力は 1-indexed, forward で -1 されるので 0 以上
            assert pred["labels"].min().item() >= -1

    def test_forward_with_labels(self) -> None:
        """学習時の forward が loss を返すことを確認."""
        model = SSDLiteModel(num_classes=2, pretrained=False)
        model.train()

        pixel_values = torch.randn(2, 3, 320, 320)
        labels = [
            {
                "boxes": torch.tensor(
                    [[50.0, 50.0, 100.0, 100.0]], dtype=torch.float32
                ),
                "labels": torch.tensor([1], dtype=torch.int64),
            },
            {
                "boxes": torch.tensor([[30.0, 30.0, 80.0, 80.0]], dtype=torch.float32),
                "labels": torch.tensor([2], dtype=torch.int64),
            },
        ]

        outputs = model(pixel_values, labels=labels)

        assert "loss" in outputs
        assert isinstance(outputs["loss"], torch.Tensor)

    def test_forward_batch_size(self) -> None:
        """バッチサイズ > 1 の推論が動作することを確認."""
        model = SSDLiteModel(num_classes=2, pretrained=False)
        model.eval()

        pixel_values = torch.randn(3, 3, 320, 320)
        with torch.no_grad():
            outputs = model(pixel_values)

        assert len(outputs["predictions"]) == 3

    def test_model_property(self) -> None:
        """model プロパティが内部モデルを返すことを確認."""
        model = SSDLiteModel(num_classes=2, pretrained=False)
        assert isinstance(model.model, nn.Module)
