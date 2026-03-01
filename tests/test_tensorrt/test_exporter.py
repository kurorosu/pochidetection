"""TensorRTExporterのテスト."""

from pathlib import Path

import pytest

pytest.importorskip("tensorrt")

from pochidetection.tensorrt import TensorRTExporter

from .conftest import INPUT_SIZE


class TestTensorRTExporter:
    """TensorRTExporterのテスト."""

    def test_init(self) -> None:
        """初期化が正常に行われることを確認."""
        exporter = TensorRTExporter()
        assert exporter is not None
        assert exporter.trt_logger is not None

    def test_export_creates_valid_file(
        self, dummy_onnx_path: Path, tmp_path: Path
    ) -> None:
        """正常にTensorRTエンジンが書き出されることを確認する."""
        exporter = TensorRTExporter()
        output_path = tmp_path / "model.engine"

        result_path = exporter.export(
            onnx_path=dummy_onnx_path,
            output_path=output_path,
            input_size=INPUT_SIZE,
            min_batch=1,
            opt_batch=1,
            max_batch=2,
        )

        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_export_invalid_onnx_path(self, tmp_path: Path) -> None:
        """存在しないONNXパスを指定した場合FileNotFoundErrorが発生することを確認."""
        exporter = TensorRTExporter()
        with pytest.raises(FileNotFoundError):
            exporter.export(
                onnx_path=tmp_path / "non_existent.onnx",
                output_path=tmp_path / "model.engine",
                input_size=INPUT_SIZE,
            )
