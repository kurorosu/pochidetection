"""OnnxBackend のテスト."""

from pathlib import Path

import numpy as np
import pytest
import torch

from pochidetection.inference import OnnxBackend
from pochidetection.models import RTDetrModel

from .conftest import INPUT_SIZE


class TestOnnxBackendInit:
    """OnnxBackend の初期化テスト."""

    def test_init_creates_session(self, onnx_path: Path) -> None:
        """正常な ONNX ファイルでセッションが作成されることを確認."""
        backend = OnnxBackend(onnx_path, providers=["CPUExecutionProvider"])
        assert backend.session is not None

    def test_init_file_not_found(self, tmp_path: Path) -> None:
        """存在しないファイルで FileNotFoundError が発生することを確認."""
        with pytest.raises(FileNotFoundError, match="ONNXモデルが見つかりません"):
            OnnxBackend(tmp_path / "nonexistent.onnx")

    def test_init_directory_raises_value_error(self, tmp_path: Path) -> None:
        """ディレクトリを指定すると ValueError が発生することを確認."""
        onnx_dir = tmp_path / "model.onnx"
        onnx_dir.mkdir()
        with pytest.raises(ValueError, match="ファイルである必要があります"):
            OnnxBackend(onnx_dir)

    def test_init_non_onnx_suffix_raises_value_error(self, tmp_path: Path) -> None:
        """.onnx 以外の拡張子で ValueError が発生することを確認."""
        non_onnx = tmp_path / "model.pt"
        non_onnx.write_text("dummy")
        with pytest.raises(ValueError, match=".onnx である必要があります"):
            OnnxBackend(non_onnx)

    def test_init_stores_input_names(self, onnx_path: Path) -> None:
        """入力名がセッションから正しく取得されることを確認."""
        backend = OnnxBackend(onnx_path, providers=["CPUExecutionProvider"])
        assert "pixel_values" in backend._input_names

    def test_init_stores_output_names(self, onnx_path: Path) -> None:
        """出力名がセッションから正しく取得されることを確認."""
        backend = OnnxBackend(onnx_path, providers=["CPUExecutionProvider"])
        assert "logits" in backend._output_names
        assert "pred_boxes" in backend._output_names


class TestOnnxBackendProviders:
    """Execution Providers のテスト."""

    def test_explicit_cpu_provider(self, onnx_path: Path) -> None:
        """CPUExecutionProvider を明示指定して動作することを確認."""
        backend = OnnxBackend(onnx_path, providers=["CPUExecutionProvider"])
        providers = backend.session.get_providers()
        assert "CPUExecutionProvider" in providers

    def test_resolve_providers_cpu_device(self) -> None:
        """device='cpu' で CPUExecutionProvider のみ返ることを確認."""
        providers = OnnxBackend._resolve_providers("cpu")
        assert providers == ["CPUExecutionProvider"]

    def test_resolve_providers_cuda_device_includes_cpu(self) -> None:
        """device='cuda' で CPUExecutionProvider が含まれることを確認."""
        providers = OnnxBackend._resolve_providers("cuda")
        assert "CPUExecutionProvider" in providers


class TestOnnxBackendInfer:
    """OnnxBackend.infer のテスト."""

    def test_infer_returns_torch_tensors(self, onnx_path: Path) -> None:
        """推論結果が torch.Tensor で返ることを確認."""
        backend = OnnxBackend(onnx_path, providers=["CPUExecutionProvider"])
        inputs = {"pixel_values": torch.randn(1, 3, *INPUT_SIZE)}

        pred_logits, pred_boxes = backend.infer(inputs)

        assert isinstance(pred_logits, torch.Tensor)
        assert isinstance(pred_boxes, torch.Tensor)

    def test_infer_output_shapes(self, onnx_path: Path) -> None:
        """推論結果の shape が正しいことを確認."""
        backend = OnnxBackend(onnx_path, providers=["CPUExecutionProvider"])
        inputs = {"pixel_values": torch.randn(1, 3, *INPUT_SIZE)}

        pred_logits, pred_boxes = backend.infer(inputs)

        assert pred_logits.ndim == 3
        assert pred_logits.shape[0] == 1
        assert pred_boxes.ndim == 3
        assert pred_boxes.shape[0] == 1
        assert pred_boxes.shape[2] == 4

    def test_infer_missing_input_raises_value_error(self, onnx_path: Path) -> None:
        """必須入力が欠けている場合 ValueError が発生することを確認."""
        backend = OnnxBackend(onnx_path, providers=["CPUExecutionProvider"])
        inputs = {"wrong_key": torch.randn(1, 3, *INPUT_SIZE)}

        with pytest.raises(ValueError, match="ONNX入力が不足しています"):
            backend.infer(inputs)

    def test_infer_extra_keys_are_ignored(self, onnx_path: Path) -> None:
        """余分なキーが含まれていても正常に動作することを確認."""
        backend = OnnxBackend(onnx_path, providers=["CPUExecutionProvider"])
        inputs = {
            "pixel_values": torch.randn(1, 3, *INPUT_SIZE),
            "extra_key": torch.randn(1),
        }

        pred_logits, pred_boxes = backend.infer(inputs)

        assert isinstance(pred_logits, torch.Tensor)
        assert isinstance(pred_boxes, torch.Tensor)


class TestOnnxBackendSynchronize:
    """OnnxBackend.synchronize のテスト."""

    def test_synchronize_does_nothing(self, onnx_path: Path) -> None:
        """synchronize が例外なく完了することを確認."""
        backend = OnnxBackend(onnx_path, providers=["CPUExecutionProvider"])
        backend.synchronize()


class TestOnnxBackendOutputEquivalence:
    """PyTorch と ONNX の出力同等性テスト."""

    def test_pytorch_onnx_output_close(
        self, rtdetr_model: RTDetrModel, onnx_path: Path
    ) -> None:
        """PyTorch と ONNX の出力が許容誤差内で一致することを確認."""
        rtdetr_model.eval()
        dummy_input = torch.randn(1, 3, *INPUT_SIZE)

        with torch.no_grad():
            pt_outputs = rtdetr_model.model(dummy_input)
        pt_logits = pt_outputs.logits.numpy()
        pt_boxes = pt_outputs.pred_boxes.numpy()

        backend = OnnxBackend(onnx_path, providers=["CPUExecutionProvider"])
        inputs = {"pixel_values": dummy_input}
        onnx_logits, onnx_boxes = backend.infer(inputs)

        assert np.allclose(pt_logits, onnx_logits.numpy(), rtol=1e-3, atol=1e-5)
        assert np.allclose(pt_boxes, onnx_boxes.numpy(), rtol=1e-3, atol=1e-5)
