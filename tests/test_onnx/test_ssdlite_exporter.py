"""SSDLiteOnnxExporter クラスのテスト."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from pochidetection.models import SSDLiteModel

pytestmark = pytest.mark.slow
from pochidetection.onnx import SSDLiteOnnxExporter

from .conftest import SSDLITE_INPUT_SIZE
from .conftest import SSDLITE_NUM_CLASSES as NUM_CLASSES


class TestSSDLiteOnnxExporterInit:
    """SSDLiteOnnxExporter 初期化のテスト."""

    def test_init_default(self) -> None:
        """デフォルト初期化の値を確認する."""
        exporter = SSDLiteOnnxExporter()
        assert exporter.model is None
        assert exporter.device == torch.device("cpu")

    def test_init_with_model(self, ssdlite_model: SSDLiteModel) -> None:
        """モデル指定初期化を確認する."""
        exporter = SSDLiteOnnxExporter(model=ssdlite_model)
        assert exporter.model is ssdlite_model


class TestSSDLiteOnnxExporterExport:
    """SSDLiteOnnxExporter.export のテスト."""

    def test_export_creates_valid_file(self, ssdlite_onnx_path: Path) -> None:
        """エクスポートされたファイルが存在しサイズが正であることを確認する."""
        assert ssdlite_onnx_path.exists()
        assert ssdlite_onnx_path.stat().st_size > 0

    def test_export_fp16_creates_smaller_file(
        self, ssdlite_onnx_path: Path, ssdlite_onnx_fp16_path: Path
    ) -> None:
        """FP16 ファイルが FP32 より小さいことを確認する."""
        fp32_size = ssdlite_onnx_path.stat().st_size
        fp16_size = ssdlite_onnx_fp16_path.stat().st_size
        assert fp16_size < fp32_size

    def test_export_without_model_raises_error(self, tmp_path: Path) -> None:
        """モデル未設定で ValueError になることを確認する."""
        exporter = SSDLiteOnnxExporter()
        with pytest.raises(ValueError, match="モデル"):
            exporter.export(tmp_path / "model.onnx", input_size=SSDLITE_INPUT_SIZE)


class TestSSDLiteOnnxExporterVerify:
    """SSDLiteOnnxExporter.verify のテスト."""

    def test_verify_fp32_returns_true(
        self, ssdlite_model: SSDLiteModel, ssdlite_onnx_path: Path
    ) -> None:
        """FP32: PyTorch と ONNX の出力が一致することを確認する."""
        exporter = SSDLiteOnnxExporter(model=ssdlite_model)
        assert exporter.verify(ssdlite_onnx_path, input_size=SSDLITE_INPUT_SIZE)

    def test_verify_fp16_returns_true(
        self, ssdlite_model: SSDLiteModel, ssdlite_onnx_fp16_path: Path
    ) -> None:
        """FP16: PyTorch と ONNX の出力が許容誤差内で一致することを確認する."""
        exporter = SSDLiteOnnxExporter(model=ssdlite_model)
        assert exporter.verify(
            ssdlite_onnx_fp16_path, input_size=SSDLITE_INPUT_SIZE, fp16=True
        )

    def test_verify_without_model_raises_error(self, ssdlite_onnx_path: Path) -> None:
        """モデル未設定で ValueError になることを確認する."""
        exporter = SSDLiteOnnxExporter()
        with pytest.raises(ValueError, match="モデル"):
            exporter.verify(ssdlite_onnx_path, input_size=SSDLITE_INPUT_SIZE)


class TestSSDLiteOnnxRuntimeInference:
    """SSDLite ONNX モデルの ONNX Runtime 推論テスト."""

    def test_onnx_runtime_inference_fp32(self, ssdlite_onnx_path: Path) -> None:
        """FP32 ダミー入力で ONNX Runtime 推論が正常に完了し出力 shape が正しいことを確認する."""
        session = ort.InferenceSession(
            str(ssdlite_onnx_path), providers=["CPUExecutionProvider"]
        )
        dummy_input = np.random.randn(1, 3, *SSDLITE_INPUT_SIZE).astype(np.float32)
        outputs = session.run(None, {"pixel_values": dummy_input})

        cls_logits, bbox_regression = outputs[0], outputs[1]
        assert cls_logits.ndim == 3
        assert cls_logits.shape[0] == 1
        # num_classes + 1 (background)
        assert cls_logits.shape[2] == NUM_CLASSES + 1
        assert bbox_regression.ndim == 3
        assert bbox_regression.shape[0] == 1
        assert bbox_regression.shape[2] == 4

    def test_onnx_runtime_inference_fp16(self, ssdlite_onnx_fp16_path: Path) -> None:
        """FP16 ダミー入力で ONNX Runtime 推論が正常に完了することを確認する."""
        session = ort.InferenceSession(
            str(ssdlite_onnx_fp16_path), providers=["CPUExecutionProvider"]
        )
        dummy_input = np.random.randn(1, 3, *SSDLITE_INPUT_SIZE).astype(np.float16)
        outputs = session.run(None, {"pixel_values": dummy_input})

        cls_logits, bbox_regression = outputs[0], outputs[1]
        assert cls_logits.ndim == 3
        assert cls_logits.shape[0] == 1
        # num_classes + 1 (background)
        assert cls_logits.shape[2] == NUM_CLASSES + 1
        assert bbox_regression.ndim == 3
        assert bbox_regression.shape[0] == 1
        assert bbox_regression.shape[2] == 4


class TestSSDLiteOnnxExporterLoadModel:
    """SSDLiteOnnxExporter.load_model のテスト."""

    def test_load_model_from_saved_directory(
        self, ssdlite_model: SSDLiteModel, tmp_path: Path
    ) -> None:
        """state_dict 形式から load_model でモデルを復元できることを検証する."""
        save_dir = tmp_path / "saved_model"
        ssdlite_model.save(save_dir)

        exporter = SSDLiteOnnxExporter()
        exporter.load_model(save_dir, num_classes=NUM_CLASSES, pretrained=False)
        assert exporter.model is not None
        assert exporter.model.num_classes == NUM_CLASSES
