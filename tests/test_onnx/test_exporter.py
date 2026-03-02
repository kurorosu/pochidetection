"""OnnxExporterクラスのテスト."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from pochidetection.models import RTDetrModel
from pochidetection.onnx import OnnxExporter
from tests.conftest import ONNX_INPUT_SIZE as INPUT_SIZE


class TestOnnxExporterInit:
    """OnnxExporter初期化のテスト."""

    def test_init_default(self) -> None:
        """デフォルト初期化の値を確認する."""
        exporter = OnnxExporter()
        assert exporter.model is None
        assert exporter.device == torch.device("cpu")

    def test_init_with_model(self, rtdetr_model: RTDetrModel) -> None:
        """モデル指定初期化を確認する."""
        exporter = OnnxExporter(model=rtdetr_model)
        assert exporter.model is rtdetr_model


class TestOnnxExporterExport:
    """OnnxExporter.exportのテスト."""

    def test_export_creates_valid_file(self, onnx_path: Path) -> None:
        """エクスポートされたファイルが存在しサイズが正であることを確認する."""
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_export_without_model_raises_error(self, tmp_path: Path) -> None:
        """モデル未設定でValueErrorになることを確認する."""
        exporter = OnnxExporter()
        with pytest.raises(ValueError, match="モデル"):
            exporter.export(tmp_path / "model.onnx", input_size=INPUT_SIZE)


class TestOnnxExporterVerify:
    """OnnxExporter.verifyのテスト."""

    def test_verify_returns_true(
        self, rtdetr_model: RTDetrModel, onnx_path: Path
    ) -> None:
        """PyTorchとONNXの出力が一致することを確認する."""
        exporter = OnnxExporter(model=rtdetr_model)
        assert exporter.verify(onnx_path, input_size=INPUT_SIZE) is True


class TestOnnxRuntimeInference:
    """ONNXモデルのONNX Runtime推論テスト."""

    def test_onnx_runtime_inference(self, onnx_path: Path) -> None:
        """ダミー入力でONNX Runtime推論が正常に完了し出力shapeが正しいことを確認する."""
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        dummy_input = np.random.randn(1, 3, *INPUT_SIZE).astype(np.float32)
        outputs = session.run(None, {"pixel_values": dummy_input})

        logits, pred_boxes = outputs[0], outputs[1]
        assert logits.ndim == 3
        assert logits.shape[0] == 1
        assert pred_boxes.ndim == 3
        assert pred_boxes.shape[0] == 1
        assert pred_boxes.shape[2] == 4


class TestOnnxExporterLoadModel:
    """OnnxExporter.load_modelのテスト."""

    def test_load_model_from_saved_directory(
        self, rtdetr_model: RTDetrModel, tmp_path: Path
    ) -> None:
        """save_pretrained形式からload_modelでモデルを復元できることを検証する."""
        save_dir = tmp_path / "saved_model"
        rtdetr_model.model.save_pretrained(save_dir)

        exporter = OnnxExporter()
        exporter.load_model(save_dir)
        assert exporter.model is not None
        assert exporter.model.num_classes == rtdetr_model.num_classes
