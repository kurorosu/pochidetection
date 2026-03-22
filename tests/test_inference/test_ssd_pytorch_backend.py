"""SsdPyTorchBackend のテスト."""

import torch

from pochidetection.inference.ssd import SsdPyTorchBackend
from pochidetection.models import SSD300Model, SSDLiteModel


class TestSsdPyTorchBackendInfer:
    """SsdPyTorchBackend.infer のテスト."""

    def test_infer_with_ssdlite_model(self, ssdlite_model: SSDLiteModel) -> None:
        """SSDLiteModel で推論結果が返ることを確認."""
        backend = SsdPyTorchBackend(ssdlite_model)

        inputs = {"pixel_values": torch.randn(1, 3, 320, 320)}
        pred = backend.infer(inputs)

        assert "boxes" in pred
        assert "scores" in pred
        assert "labels" in pred

    def test_infer_with_ssd300_model(self, ssd300_model: SSD300Model) -> None:
        """SSD300Model で推論結果が返ることを確認."""
        backend = SsdPyTorchBackend(ssd300_model)

        inputs = {"pixel_values": torch.randn(1, 3, 300, 300)}
        pred = backend.infer(inputs)

        assert "boxes" in pred
        assert "scores" in pred
        assert "labels" in pred

    def test_infer_output_tensors(self, ssd300_model: SSD300Model) -> None:
        """推論結果の各キーが Tensor であることを確認."""
        backend = SsdPyTorchBackend(ssd300_model)

        inputs = {"pixel_values": torch.randn(1, 3, 300, 300)}
        pred = backend.infer(inputs)

        assert isinstance(pred["boxes"], torch.Tensor)
        assert isinstance(pred["scores"], torch.Tensor)
        assert isinstance(pred["labels"], torch.Tensor)
