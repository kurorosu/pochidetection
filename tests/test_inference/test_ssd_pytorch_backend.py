"""SsdPyTorchBackend のテスト."""

from unittest.mock import MagicMock, patch

import torch

from pochidetection.inference.ssd import SsdPyTorchBackend
from pochidetection.models import SSD300Model, SSDLiteModel


class TestSsdPyTorchBackendInfer:
    """SsdPyTorchBackend.infer のテスト."""

    def test_infer_with_ssdlite_model(self) -> None:
        """SSDLiteModel で推論結果が返ることを確認."""
        model = SSDLiteModel(num_classes=2, pretrained=False)
        model.eval()
        backend = SsdPyTorchBackend(model)

        inputs = {"pixel_values": torch.randn(1, 3, 320, 320)}
        pred = backend.infer(inputs)

        assert "boxes" in pred
        assert "scores" in pred
        assert "labels" in pred

    def test_infer_with_ssd300_model(self) -> None:
        """SSD300Model で推論結果が返ることを確認."""
        model = SSD300Model(num_classes=2, pretrained=False)
        model.eval()
        backend = SsdPyTorchBackend(model)

        inputs = {"pixel_values": torch.randn(1, 3, 300, 300)}
        pred = backend.infer(inputs)

        assert "boxes" in pred
        assert "scores" in pred
        assert "labels" in pred

    def test_infer_output_tensors(self) -> None:
        """推論結果の各キーが Tensor であることを確認."""
        model = SSD300Model(num_classes=2, pretrained=False)
        model.eval()
        backend = SsdPyTorchBackend(model)

        inputs = {"pixel_values": torch.randn(1, 3, 300, 300)}
        pred = backend.infer(inputs)

        assert isinstance(pred["boxes"], torch.Tensor)
        assert isinstance(pred["scores"], torch.Tensor)
        assert isinstance(pred["labels"], torch.Tensor)


class TestSsdPyTorchBackendSynchronize:
    """SsdPyTorchBackend.synchronize のテスト."""

    def test_synchronize_on_cpu_does_nothing(self) -> None:
        """CPU デバイスでは synchronize が例外なく完了することを確認."""
        model = SSD300Model(num_classes=2, pretrained=False)
        model.eval()
        backend = SsdPyTorchBackend(model)

        backend.synchronize()

    @patch("pochidetection.inference.ssd.pytorch_backend.synchronize_cuda")
    def test_synchronize_delegates_to_sync_helper(self, mock_sync: MagicMock) -> None:
        """synchronize() が synchronize_cuda に委譲することを確認."""
        model = SSD300Model(num_classes=2, pretrained=False)
        model.eval()
        backend = SsdPyTorchBackend(model)

        backend.synchronize()

        mock_sync.assert_called_once_with(model)
