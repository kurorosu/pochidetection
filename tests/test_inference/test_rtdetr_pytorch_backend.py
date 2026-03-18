"""RTDetrPyTorchBackend のテスト."""

import pytest
import torch

from pochidetection.inference import RTDetrPyTorchBackend
from pochidetection.models import RTDetrModel


@pytest.fixture()
def backend(rtdetr_model: RTDetrModel) -> RTDetrPyTorchBackend:
    """テスト用 RTDetrPyTorchBackend."""
    return RTDetrPyTorchBackend(rtdetr_model)


class TestRTDetrPyTorchBackendInfer:
    """RTDetrPyTorchBackend.infer のテスト."""

    def test_infer_returns_logits_and_boxes(
        self, backend: RTDetrPyTorchBackend
    ) -> None:
        """推論結果が logits と pred_boxes のタプルで返ることを確認."""
        inputs = {"pixel_values": torch.randn(1, 3, 64, 64)}

        with torch.no_grad():
            logits, boxes = backend.infer(inputs)

        assert isinstance(logits, torch.Tensor)
        assert isinstance(boxes, torch.Tensor)

    def test_infer_output_shapes(self, backend: RTDetrPyTorchBackend) -> None:
        """推論結果の shape が正しいことを確認."""
        inputs = {"pixel_values": torch.randn(1, 3, 64, 64)}

        with torch.no_grad():
            logits, boxes = backend.infer(inputs)

        assert logits.ndim == 3
        assert logits.shape[0] == 1
        assert logits.shape[2] == 2  # num_classes
        assert boxes.ndim == 3
        assert boxes.shape[0] == 1
        assert boxes.shape[2] == 4

    def test_infer_num_queries(self, backend: RTDetrPyTorchBackend) -> None:
        """出力のクエリ数が config と一致することを確認."""
        inputs = {"pixel_values": torch.randn(1, 3, 64, 64)}

        with torch.no_grad():
            logits, boxes = backend.infer(inputs)

        assert logits.shape[1] == 50
        assert boxes.shape[1] == 50
