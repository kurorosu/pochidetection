"""SSDLiteModel のテスト."""

import pytest
import torch
import torch.nn as nn

from pochidetection.interfaces import IDetectionModel
from pochidetection.models import SSDLiteModel


class TestSSDLiteModel:
    """SSDLiteModel のテスト."""

    def test_implements_interface(self, ssdlite_model: SSDLiteModel) -> None:
        """IDetectionModel を実装していることを確認."""
        assert isinstance(ssdlite_model, IDetectionModel)

    def test_is_nn_module(self, ssdlite_model: SSDLiteModel) -> None:
        """nn.Module を継承していることを確認."""
        assert isinstance(ssdlite_model, nn.Module)

    @pytest.mark.slow
    def test_num_classes_property(self) -> None:
        """num_classes がユーザ指定値 (背景なし) を返すことを確認."""
        model = SSDLiteModel(num_classes=4, pretrained=False)
        assert model.num_classes == 4

    def test_forward_inference(self, ssdlite_model: SSDLiteModel) -> None:
        """推論時の forward が predictions を返すことを確認."""
        pixel_values = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            outputs = ssdlite_model(pixel_values)

        assert "predictions" in outputs
        assert len(outputs["predictions"]) == 1

        pred = outputs["predictions"][0]
        assert "boxes" in pred
        assert "scores" in pred
        assert "labels" in pred

    def test_forward_inference_labels_are_0_indexed(
        self, ssdlite_model: SSDLiteModel
    ) -> None:
        """推論時のラベルが 0-indexed (>= 0) であることを確認."""
        pixel_values = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            outputs = ssdlite_model(pixel_values)

        pred = outputs["predictions"][0]
        if pred["labels"].numel() > 0:
            # 背景クラスは除去済みなので全て 0 以上
            assert pred["labels"].min().item() >= 0

    def test_forward_with_labels(self, ssdlite_model: SSDLiteModel) -> None:
        """学習時の forward が loss を返すことを確認 (0-indexed ラベル)."""
        was_training = ssdlite_model.training
        ssdlite_model.train()

        pixel_values = torch.randn(2, 3, 320, 320)
        labels = [
            {
                "boxes": torch.tensor(
                    [[50.0, 50.0, 100.0, 100.0]], dtype=torch.float32
                ),
                "class_labels": torch.tensor([0], dtype=torch.int64),
            },
            {
                "boxes": torch.tensor([[30.0, 30.0, 80.0, 80.0]], dtype=torch.float32),
                "class_labels": torch.tensor([1], dtype=torch.int64),
            },
        ]

        outputs = ssdlite_model(pixel_values, labels=labels)

        assert "loss" in outputs
        assert isinstance(outputs["loss"], torch.Tensor)

        if not was_training:
            ssdlite_model.eval()

    def test_forward_batch_size(self, ssdlite_model: SSDLiteModel) -> None:
        """バッチサイズ > 1 の推論が動作することを確認."""
        pixel_values = torch.randn(3, 3, 320, 320)
        with torch.no_grad():
            outputs = ssdlite_model(pixel_values)

        assert len(outputs["predictions"]) == 3

    def test_model_property(self, ssdlite_model: SSDLiteModel) -> None:
        """model プロパティが内部モデルを返すことを確認."""
        assert isinstance(ssdlite_model.model, nn.Module)

    def test_nms_iou_threshold_default(self, ssdlite_model: SSDLiteModel) -> None:
        """デフォルトの nms_iou_threshold が torchvision に渡されることを確認."""
        assert ssdlite_model.model.nms_thresh == 0.55

    @pytest.mark.slow
    def test_nms_iou_threshold_custom(self) -> None:
        """カスタム nms_iou_threshold が torchvision に渡されることを確認."""
        model = SSDLiteModel(num_classes=2, pretrained=False, nms_iou_threshold=0.3)
        assert model.model.nms_thresh == 0.3
