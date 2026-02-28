"""RTDetrModelのテスト."""

import pytest
import torch
import torch.nn as nn

from pochidetection.interfaces import IDetectionModel
from pochidetection.models import RTDetrModel


class TestRTDetrModel:
    """RTDetrModelのテスト."""

    @pytest.fixture
    def model(self) -> RTDetrModel:
        """テスト用モデルを作成するfixture (推論用).

        Note:
            実際のモデルをダウンロードするため, CIでは時間がかかる可能性がある.
        """
        return RTDetrModel(model_name="PekingU/rtdetr_r18vd", num_classes=2)

    @pytest.fixture
    def model_for_training(self) -> RTDetrModel:
        """学習テスト用モデルを作成するfixture.

        Note:
            ラベル付きforwardには, 事前学習なしでconfigからモデルを作成する必要がある.
        """
        return RTDetrModel(
            model_name="PekingU/rtdetr_r18vd", num_classes=2, pretrained=False
        )

    def test_implements_interface(self, model: RTDetrModel) -> None:
        """IDetectionModelを実装していることを確認."""
        assert isinstance(model, IDetectionModel)

    def test_is_nn_module(self, model: RTDetrModel) -> None:
        """nn.Moduleを継承していることを確認."""
        assert isinstance(model, nn.Module)

    def test_forward_inference(self, model: RTDetrModel) -> None:
        """推論時のforward処理が正しく動作することを確認."""
        model.eval()
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 640, 640)

        with torch.no_grad():
            outputs = model(pixel_values)

        assert "pred_logits" in outputs
        assert "pred_boxes" in outputs
        assert outputs["pred_logits"].shape[0] == batch_size
        assert outputs["pred_boxes"].shape[0] == batch_size
        assert outputs["pred_boxes"].shape[2] == 4  # [cx, cy, w, h]

    def test_forward_with_labels(self, model_for_training: RTDetrModel) -> None:
        """学習時のforward処理が損失を返すことを確認."""
        model_for_training.train()
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 640, 640)

        # ラベルを作成 (正規化座標)
        labels = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
                "class_labels": torch.tensor([0], dtype=torch.int64),
            },
            {
                "boxes": torch.tensor(
                    [[0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.15, 0.15]], dtype=torch.float32
                ),
                "class_labels": torch.tensor([1, 0], dtype=torch.int64),
            },
        ]

        outputs = model_for_training(pixel_values, labels=labels)

        assert "loss" in outputs
        assert "pred_logits" in outputs
        assert "pred_boxes" in outputs
        assert isinstance(outputs["loss"], torch.Tensor)

    def test_get_backbone_params(self, model: RTDetrModel) -> None:
        """バックボーンパラメータを取得できることを確認."""
        backbone_params = model.get_backbone_params()

        assert isinstance(backbone_params, list)
        assert len(backbone_params) > 0
        assert all(isinstance(p, nn.Parameter) for p in backbone_params)

    def test_get_head_params(self, model: RTDetrModel) -> None:
        """ヘッドパラメータを取得できることを確認."""
        head_params = model.get_head_params()

        assert isinstance(head_params, list)
        assert len(head_params) > 0
        assert all(isinstance(p, nn.Parameter) for p in head_params)

    def test_backbone_head_params_disjoint(self, model: RTDetrModel) -> None:
        """バックボーンとヘッドのパラメータが重複しないことを確認."""
        backbone_params = set(id(p) for p in model.get_backbone_params())
        head_params = set(id(p) for p in model.get_head_params())

        assert backbone_params.isdisjoint(head_params)

    def test_all_params_covered(self, model: RTDetrModel) -> None:
        """全パラメータがバックボーンまたはヘッドに含まれることを確認."""
        backbone_params = set(id(p) for p in model.get_backbone_params())
        head_params = set(id(p) for p in model.get_head_params())
        all_params = set(id(p) for p in model.parameters())

        covered = backbone_params | head_params
        assert covered == all_params

    def test_num_classes_property(self, model: RTDetrModel) -> None:
        """num_classesプロパティが正しい値を返すことを確認."""
        assert model.num_classes == 2

    def test_model_property(self, model: RTDetrModel) -> None:
        """modelプロパティが内部モデルを返すことを確認."""
        from transformers import RTDetrForObjectDetection

        assert isinstance(model.model, RTDetrForObjectDetection)

    def test_custom_num_classes(self) -> None:
        """カスタムクラス数でモデルを初期化できることを確認."""
        model = RTDetrModel(
            model_name="PekingU/rtdetr_r18vd", num_classes=10, pretrained=False
        )
        assert model.num_classes == 10

        # 分類ヘッドの出力次元を確認
        model.eval()
        pixel_values = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(pixel_values)

        assert outputs["pred_logits"].shape[2] == 10
