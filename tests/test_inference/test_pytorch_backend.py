"""PyTorchBackend のテスト."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from pochidetection.inference import PyTorchBackend
from pochidetection.models import RTDetrModel


@pytest.fixture()
def backend(rtdetr_model: RTDetrModel) -> PyTorchBackend:
    """テスト用 PyTorchBackend."""
    return PyTorchBackend(rtdetr_model)


class TestPyTorchBackendInfer:
    """PyTorchBackend.infer のテスト."""

    def test_infer_returns_logits_and_boxes(self, backend: PyTorchBackend) -> None:
        """推論結果が logits と pred_boxes のタプルで返ることを確認."""
        inputs = {"pixel_values": torch.randn(1, 3, 64, 64)}

        with torch.no_grad():
            logits, boxes = backend.infer(inputs)

        assert isinstance(logits, torch.Tensor)
        assert isinstance(boxes, torch.Tensor)

    def test_infer_output_shapes(self, backend: PyTorchBackend) -> None:
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

    def test_infer_num_queries(self, backend: PyTorchBackend) -> None:
        """出力のクエリ数が config と一致することを確認."""
        inputs = {"pixel_values": torch.randn(1, 3, 64, 64)}

        with torch.no_grad():
            logits, boxes = backend.infer(inputs)

        assert logits.shape[1] == 50
        assert boxes.shape[1] == 50


class TestPyTorchBackendSynchronize:
    """PyTorchBackend.synchronize のテスト."""

    def test_synchronize_on_cpu_does_nothing(self, backend: PyTorchBackend) -> None:
        """CPU デバイスでは synchronize が例外なく完了することを確認."""
        backend.synchronize()

    @patch("pochidetection.inference.pytorch_backend.torch.cuda.is_available")
    def test_synchronize_skips_when_cuda_unavailable(
        self, mock_is_available: MagicMock, backend: PyTorchBackend
    ) -> None:
        """CUDA が利用不可の場合, synchronize() を呼び出さないことを確認."""
        mock_is_available.return_value = False

        with patch(
            "pochidetection.inference.pytorch_backend.torch.cuda.synchronize"
        ) as mock_sync:
            backend.synchronize()
            mock_sync.assert_not_called()
