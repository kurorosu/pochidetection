"""TensorRTExporterのテスト."""

from pathlib import Path

import pytest

pytest.importorskip("tensorrt")

from pochidetection.tensorrt import INT8Calibrator, TensorRTExporter

from .conftest import INPUT_SIZE

pytestmark = pytest.mark.slow


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

    def test_export_fp16_creates_valid_file(
        self, dummy_onnx_path: Path, tmp_path: Path
    ) -> None:
        """FP16モードでTensorRTエンジンが正常に書き出されることを確認する."""
        exporter = TensorRTExporter()
        output_path = tmp_path / "model_fp16.engine"

        result_path = exporter.export(
            onnx_path=dummy_onnx_path,
            output_path=output_path,
            input_size=INPUT_SIZE,
            min_batch=1,
            opt_batch=1,
            max_batch=2,
            use_fp16=True,
        )

        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_export_fp16_fallback_logs_warning(
        self, dummy_onnx_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FP16非対応GPUで警告ログが出力されFP32にフォールバックすることを確認."""
        import tensorrt as trt

        original_init = trt.Builder.__init__

        def patched_init(self_builder: trt.Builder, *args: object) -> None:
            original_init(self_builder, *args)
            monkeypatch.setattr(
                type(self_builder), "platform_has_fast_fp16", property(lambda _: False)
            )

        monkeypatch.setattr(trt.Builder, "__init__", patched_init)

        exporter = TensorRTExporter()
        output_path = tmp_path / "model_fallback.engine"

        result_path = exporter.export(
            onnx_path=dummy_onnx_path,
            output_path=output_path,
            input_size=INPUT_SIZE,
            min_batch=1,
            opt_batch=1,
            max_batch=2,
            use_fp16=True,
        )

        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_export_int8_creates_valid_file(
        self,
        dummy_onnx_path: Path,
        calib_image_dir: Path,
        tmp_path: Path,
    ) -> None:
        """INT8モードでTensorRTエンジンが正常に書き出されることを確認する."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
        )
        exporter = TensorRTExporter()
        output_path = tmp_path / "model_int8.engine"

        result_path = exporter.export(
            onnx_path=dummy_onnx_path,
            output_path=output_path,
            input_size=INPUT_SIZE,
            min_batch=1,
            opt_batch=1,
            max_batch=2,
            use_int8=True,
            int8_calibrator=calibrator,
        )

        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_export_int8_fallback_logs_warning(
        self, dummy_onnx_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """INT8非対応GPUで警告ログが出力されFP32にフォールバックすることを確認."""
        import tensorrt as trt

        original_init = trt.Builder.__init__

        def patched_init(self_builder: trt.Builder, *args: object) -> None:
            original_init(self_builder, *args)
            monkeypatch.setattr(
                type(self_builder), "platform_has_fast_int8", property(lambda _: False)
            )

        monkeypatch.setattr(trt.Builder, "__init__", patched_init)

        exporter = TensorRTExporter()
        output_path = tmp_path / "model_int8_fallback.engine"

        result_path = exporter.export(
            onnx_path=dummy_onnx_path,
            output_path=output_path,
            input_size=INPUT_SIZE,
            min_batch=1,
            opt_batch=1,
            max_batch=2,
            use_int8=True,
        )

        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_export_int8_takes_priority_over_fp16(
        self,
        dummy_onnx_path: Path,
        calib_image_dir: Path,
        tmp_path: Path,
    ) -> None:
        """use_int8 と use_fp16 を同時指定した場合に INT8 が優先されることを確認."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
        )
        exporter = TensorRTExporter()
        output_path = tmp_path / "model_int8_priority.engine"

        result_path = exporter.export(
            onnx_path=dummy_onnx_path,
            output_path=output_path,
            input_size=INPUT_SIZE,
            min_batch=1,
            opt_batch=1,
            max_batch=2,
            use_int8=True,
            use_fp16=True,
            int8_calibrator=calibrator,
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
