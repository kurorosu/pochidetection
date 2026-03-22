"""SSDLite TensorRT エクスポートのテスト."""

from pathlib import Path

import pytest

pytest.importorskip("tensorrt")

from pochidetection.tensorrt import TensorRTExporter

from .conftest import SSDLITE_INPUT_SIZE

pytestmark = pytest.mark.slow


class TestSSDLiteTensorRTExport:
    """SSDLite ONNX モデルから TensorRT エンジンへのエクスポートテスト."""

    def test_export_fp32_creates_valid_file(
        self, ssdlite_dummy_onnx_path: Path, tmp_path: Path
    ) -> None:
        """FP32 で TensorRT エンジンが正常に書き出されることを確認."""
        exporter = TensorRTExporter()
        output_path = tmp_path / "ssdlite_fp32.engine"

        result_path = exporter.export(
            onnx_path=ssdlite_dummy_onnx_path,
            output_path=output_path,
            input_size=SSDLITE_INPUT_SIZE,
            min_batch=1,
            opt_batch=1,
            max_batch=2,
        )

        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_export_fp16_creates_valid_file(
        self, ssdlite_dummy_onnx_path: Path, tmp_path: Path
    ) -> None:
        """FP16 で TensorRT エンジンが正常に書き出されることを確認."""
        exporter = TensorRTExporter()
        output_path = tmp_path / "ssdlite_fp16.engine"

        result_path = exporter.export(
            onnx_path=ssdlite_dummy_onnx_path,
            output_path=output_path,
            input_size=SSDLITE_INPUT_SIZE,
            min_batch=1,
            opt_batch=1,
            max_batch=2,
            use_fp16=True,
        )

        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_export_invalid_onnx_path(self, tmp_path: Path) -> None:
        """存在しない ONNX パスで FileNotFoundError が発生することを確認."""
        exporter = TensorRTExporter()
        with pytest.raises(FileNotFoundError):
            exporter.export(
                onnx_path=tmp_path / "non_existent.onnx",
                output_path=tmp_path / "model.engine",
                input_size=SSDLITE_INPUT_SIZE,
            )
