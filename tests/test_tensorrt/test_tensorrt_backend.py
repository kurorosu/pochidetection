"""TensorRTBackend のテスト."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("tensorrt")

from pochidetection.inference import TensorRTBackend
from pochidetection.tensorrt.memory import TensorBinding

from .conftest import INPUT_SIZE


class TestTensorRTBackendInit:
    """TensorRTBackend の初期化テスト."""

    def test_init_creates_engine(self, engine_path: Path) -> None:
        """正常なエンジンファイルでインスタンスが作成されることを確認."""
        backend = TensorRTBackend(engine_path)
        assert backend.engine is not None

    def test_init_creates_bindings(self, engine_path: Path) -> None:
        """バインディングが正しく作成されることを確認."""
        backend = TensorRTBackend(engine_path)
        assert len(backend.bindings) > 0
        assert all(isinstance(b, TensorBinding) for b in backend.bindings)

    def test_init_file_not_found(self, tmp_path: Path) -> None:
        """存在しないファイルで FileNotFoundError が発生することを確認."""
        with pytest.raises(FileNotFoundError, match="TensorRTエンジンが見つかりません"):
            TensorRTBackend(tmp_path / "nonexistent.engine")

    def test_init_directory_raises_value_error(self, tmp_path: Path) -> None:
        """ディレクトリを指定すると ValueError が発生することを確認."""
        engine_dir = tmp_path / "model.engine"
        engine_dir.mkdir()
        with pytest.raises(ValueError, match="ファイルである必要があります"):
            TensorRTBackend(engine_dir)

    def test_init_non_engine_suffix_raises_value_error(self, tmp_path: Path) -> None:
        """.engine 以外の拡張子で ValueError が発生することを確認."""
        non_engine = tmp_path / "model.onnx"
        non_engine.write_text("dummy")
        with pytest.raises(ValueError, match=".engine である必要があります"):
            TensorRTBackend(non_engine)

    def test_init_accepts_str_path(self, engine_path: Path) -> None:
        """文字列パスでも正常に初期化できることを確認."""
        backend = TensorRTBackend(str(engine_path))
        assert backend.engine is not None


class TestTensorRTBackendInfer:
    """TensorRTBackend.infer のテスト."""

    def test_infer_returns_torch_tensors(self, engine_path: Path) -> None:
        """推論結果が torch.Tensor で返ることを確認."""
        backend = TensorRTBackend(engine_path)
        inputs = {"pixel_values": torch.randn(1, 3, *INPUT_SIZE)}

        pred_logits, pred_boxes = backend.infer(inputs)

        assert isinstance(pred_logits, torch.Tensor)
        assert isinstance(pred_boxes, torch.Tensor)

    def test_infer_output_on_cuda(self, engine_path: Path) -> None:
        """推論結果が CUDA テンソルで返ることを確認."""
        backend = TensorRTBackend(engine_path)
        inputs = {"pixel_values": torch.randn(1, 3, *INPUT_SIZE)}

        pred_logits, pred_boxes = backend.infer(inputs)

        assert pred_logits.is_cuda
        assert pred_boxes.is_cuda

    def test_infer_output_batch_size(self, engine_path: Path) -> None:
        """出力のバッチサイズが1であることを確認."""
        backend = TensorRTBackend(engine_path)
        inputs = {"pixel_values": torch.randn(1, 3, *INPUT_SIZE)}

        pred_logits, pred_boxes = backend.infer(inputs)

        assert pred_logits.shape[0] == 1
        assert pred_boxes.shape[0] == 1

    def test_infer_accepts_cpu_input(self, engine_path: Path) -> None:
        """CPU テンソルを入力しても正常に動作することを確認."""
        backend = TensorRTBackend(engine_path)
        inputs = {"pixel_values": torch.randn(1, 3, *INPUT_SIZE, device="cpu")}

        pred_logits, pred_boxes = backend.infer(inputs)

        assert isinstance(pred_logits, torch.Tensor)
        assert isinstance(pred_boxes, torch.Tensor)

    def test_infer_accepts_cuda_input(self, engine_path: Path) -> None:
        """CUDA テンソルを入力しても正常に動作することを確認."""
        backend = TensorRTBackend(engine_path)
        inputs = {"pixel_values": torch.randn(1, 3, *INPUT_SIZE, device="cuda")}

        pred_logits, pred_boxes = backend.infer(inputs)

        assert isinstance(pred_logits, torch.Tensor)
        assert isinstance(pred_boxes, torch.Tensor)

    def test_infer_missing_input_raises_value_error(self, engine_path: Path) -> None:
        """必須入力が欠けている場合 ValueError が発生することを確認."""
        backend = TensorRTBackend(engine_path)
        inputs = {"wrong_key": torch.randn(1, 3, *INPUT_SIZE)}

        with pytest.raises(ValueError, match="TensorRT入力が不足しています"):
            backend.infer(inputs)


class TestTensorRTBackendSynchronize:
    """TensorRTBackend.synchronize のテスト."""

    def test_synchronize_completes_without_error(self, engine_path: Path) -> None:
        """synchronize が例外なく完了することを確認."""
        backend = TensorRTBackend(engine_path)
        backend.synchronize()
