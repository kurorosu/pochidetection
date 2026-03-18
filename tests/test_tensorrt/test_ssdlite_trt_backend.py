"""SSDLiteTensorRTBackend のテスト."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("tensorrt")

from pochidetection.inference import SSDLiteTensorRTBackend
from pochidetection.tensorrt.memory import TensorBinding

from .conftest import SSDLITE_INPUT_SIZE, SSDLITE_NUM_CLASSES


class TestSSDLiteTensorRTBackendInit:
    """SSDLiteTensorRTBackend の初期化テスト."""

    def test_init_creates_engine(self, ssdlite_engine_path: Path) -> None:
        """正常なエンジンファイルでインスタンスが作成されることを確認."""
        backend = SSDLiteTensorRTBackend(
            ssdlite_engine_path,
            num_classes=SSDLITE_NUM_CLASSES,
            image_size=SSDLITE_INPUT_SIZE,
        )
        assert backend.engine is not None

    def test_init_creates_bindings(self, ssdlite_engine_path: Path) -> None:
        """バインディングが正しく作成されることを確認."""
        backend = SSDLiteTensorRTBackend(
            ssdlite_engine_path,
            num_classes=SSDLITE_NUM_CLASSES,
            image_size=SSDLITE_INPUT_SIZE,
        )
        assert len(backend.bindings) > 0
        assert all(isinstance(b, TensorBinding) for b in backend.bindings)

    def test_init_file_not_found(self, tmp_path: Path) -> None:
        """存在しないファイルで FileNotFoundError が発生することを確認."""
        with pytest.raises(FileNotFoundError, match="TensorRTエンジンが見つかりません"):
            SSDLiteTensorRTBackend(
                tmp_path / "nonexistent.engine",
                num_classes=SSDLITE_NUM_CLASSES,
                image_size=SSDLITE_INPUT_SIZE,
            )

    def test_init_directory_raises_value_error(self, tmp_path: Path) -> None:
        """ディレクトリを指定すると ValueError が発生することを確認."""
        engine_dir = tmp_path / "model.engine"
        engine_dir.mkdir()
        with pytest.raises(ValueError, match="ファイルである必要があります"):
            SSDLiteTensorRTBackend(
                engine_dir,
                num_classes=SSDLITE_NUM_CLASSES,
                image_size=SSDLITE_INPUT_SIZE,
            )

    def test_init_non_engine_suffix_raises_value_error(self, tmp_path: Path) -> None:
        """.engine 以外の拡張子で ValueError が発生することを確認."""
        non_engine = tmp_path / "model.onnx"
        non_engine.write_text("dummy")
        with pytest.raises(ValueError, match=".engine である必要があります"):
            SSDLiteTensorRTBackend(
                non_engine,
                num_classes=SSDLITE_NUM_CLASSES,
                image_size=SSDLITE_INPUT_SIZE,
            )

    def test_init_accepts_str_path(self, ssdlite_engine_path: Path) -> None:
        """文字列パスでも正常に初期化できることを確認."""
        backend = SSDLiteTensorRTBackend(
            str(ssdlite_engine_path),
            num_classes=SSDLITE_NUM_CLASSES,
            image_size=SSDLITE_INPUT_SIZE,
        )
        assert backend.engine is not None

    def test_init_stores_output_names(self, ssdlite_engine_path: Path) -> None:
        """出力バインディングが名前ベースで取得できることを確認."""
        backend = SSDLiteTensorRTBackend(
            ssdlite_engine_path,
            num_classes=SSDLITE_NUM_CLASSES,
            image_size=SSDLITE_INPUT_SIZE,
        )
        output_names = set(backend._output_bindings_by_name.keys())
        assert "cls_logits" in output_names
        assert "bbox_regression" in output_names


class TestSSDLiteTensorRTBackendInfer:
    """SSDLiteTensorRTBackend.infer のテスト."""

    def test_infer_returns_dict_with_expected_keys(
        self, ssdlite_engine_path: Path
    ) -> None:
        """推論結果が boxes, scores, labels キーを含む dict で返ることを確認."""
        backend = SSDLiteTensorRTBackend(
            ssdlite_engine_path,
            num_classes=SSDLITE_NUM_CLASSES,
            image_size=SSDLITE_INPUT_SIZE,
        )
        inputs = {"pixel_values": torch.randn(1, 3, *SSDLITE_INPUT_SIZE)}

        result = backend.infer(inputs)

        assert isinstance(result, dict)
        assert "boxes" in result
        assert "scores" in result
        assert "labels" in result

    def test_infer_output_shapes_consistent(self, ssdlite_engine_path: Path) -> None:
        """boxes, scores, labels の要素数が一致することを確認."""
        backend = SSDLiteTensorRTBackend(
            ssdlite_engine_path,
            num_classes=SSDLITE_NUM_CLASSES,
            image_size=SSDLITE_INPUT_SIZE,
        )
        inputs = {"pixel_values": torch.randn(1, 3, *SSDLITE_INPUT_SIZE)}

        result = backend.infer(inputs)

        num_detections = result["scores"].shape[0]
        assert result["boxes"].shape == (num_detections, 4)
        assert result["labels"].shape == (num_detections,)

    def test_infer_accepts_cpu_input(self, ssdlite_engine_path: Path) -> None:
        """CPU テンソルを入力しても正常に動作することを確認."""
        backend = SSDLiteTensorRTBackend(
            ssdlite_engine_path,
            num_classes=SSDLITE_NUM_CLASSES,
            image_size=SSDLITE_INPUT_SIZE,
        )
        inputs = {"pixel_values": torch.randn(1, 3, *SSDLITE_INPUT_SIZE, device="cpu")}

        result = backend.infer(inputs)

        assert isinstance(result, dict)
        assert "boxes" in result

    def test_infer_accepts_cuda_input(self, ssdlite_engine_path: Path) -> None:
        """CUDA テンソルを入力しても正常に動作することを確認."""
        backend = SSDLiteTensorRTBackend(
            ssdlite_engine_path,
            num_classes=SSDLITE_NUM_CLASSES,
            image_size=SSDLITE_INPUT_SIZE,
        )
        inputs = {"pixel_values": torch.randn(1, 3, *SSDLITE_INPUT_SIZE, device="cuda")}

        result = backend.infer(inputs)

        assert isinstance(result, dict)
        assert "boxes" in result

    def test_infer_missing_input_raises_value_error(
        self, ssdlite_engine_path: Path
    ) -> None:
        """必須入力が欠けている場合 ValueError が発生することを確認."""
        backend = SSDLiteTensorRTBackend(
            ssdlite_engine_path,
            num_classes=SSDLITE_NUM_CLASSES,
            image_size=SSDLITE_INPUT_SIZE,
        )
        inputs = {"wrong_key": torch.randn(1, 3, *SSDLITE_INPUT_SIZE)}

        with pytest.raises(ValueError, match="TensorRT入力が不足しています"):
            backend.infer(inputs)
