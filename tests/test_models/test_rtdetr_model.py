"""RTDetrModelのテスト."""

import torch
import torch.nn as nn

from pochidetection.interfaces import IDetectionModel
from pochidetection.models import RTDetrModel


class TestRTDetrModel:
    """RTDetrModelのテスト.

    Note:
        rtdetr_model fixtureはルートconftest.pyでsessionスコープとして定義.
    """

    def test_implements_interface(self, rtdetr_model: RTDetrModel) -> None:
        """IDetectionModelを実装していることを確認."""
        assert isinstance(rtdetr_model, IDetectionModel)

    def test_is_nn_module(self, rtdetr_model: RTDetrModel) -> None:
        """nn.Moduleを継承していることを確認."""
        assert isinstance(rtdetr_model, nn.Module)

    def test_forward_inference(self, rtdetr_model: RTDetrModel) -> None:
        """推論時のforward処理が正しく動作することを確認."""
        was_training = rtdetr_model.training
        rtdetr_model.eval()
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 64, 64)

        with torch.no_grad():
            outputs = rtdetr_model(pixel_values)

        assert "pred_logits" in outputs
        assert "pred_boxes" in outputs
        assert outputs["pred_logits"].shape[0] == batch_size
        assert outputs["pred_boxes"].shape[0] == batch_size
        assert outputs["pred_boxes"].shape[2] == 4  # [cx, cy, w, h]

        rtdetr_model.train(was_training)

    def test_forward_with_labels(self, rtdetr_model: RTDetrModel) -> None:
        """学習時のforward処理が損失を返すことを確認."""
        was_training = rtdetr_model.training
        rtdetr_model.train()
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 64, 64)

        labels = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
                "class_labels": torch.tensor([0], dtype=torch.int64),
            }
        ]

        outputs = rtdetr_model(pixel_values, labels=labels)

        assert "loss" in outputs
        assert "pred_logits" in outputs
        assert "pred_boxes" in outputs
        assert isinstance(outputs["loss"], torch.Tensor)

        rtdetr_model.train(was_training)

    def test_num_classes_property(self, rtdetr_model: RTDetrModel) -> None:
        """num_classesプロパティが正しい値を返すことを確認."""
        assert rtdetr_model.num_classes == 2

    def test_model_property(self, rtdetr_model: RTDetrModel) -> None:
        """modelプロパティが内部モデルを返すことを確認."""
        from transformers import RTDetrForObjectDetection

        assert isinstance(rtdetr_model.model, RTDetrForObjectDetection)

    def test_custom_num_classes(self) -> None:
        """カスタムクラス数でモデルを初期化できることを確認."""
        model = RTDetrModel(
            model_name="PekingU/rtdetr_r18vd", num_classes=10, pretrained=False
        )
        model.model.config.num_queries = 50

        assert model.num_classes == 10

        model.eval()
        pixel_values = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            outputs = model(pixel_values)

        assert outputs["pred_logits"].shape[2] == 10
