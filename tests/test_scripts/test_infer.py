"""infer.py のヘルパー関数テスト."""

import json
from pathlib import Path

import pytest

from pochidetection.scripts.common.inference import is_onnx_model
from pochidetection.scripts.rtdetr.infer import _load_processor


class TestIsOnnxModel:
    """is_onnx_model のテスト."""

    def test_onnx_suffix(self) -> None:
        """.onnx ファイルで True を返すことを確認."""
        assert is_onnx_model(Path("model.onnx")) is True

    def test_onnx_suffix_uppercase(self) -> None:
        """.ONNX (大文字) でも True を返すことを確認."""
        assert is_onnx_model(Path("model.ONNX")) is True

    def test_directory_path(self) -> None:
        """ディレクトリパスで False を返すことを確認."""
        assert is_onnx_model(Path("work_dirs/20260124_001/best")) is False

    def test_pt_suffix(self) -> None:
        """.pt ファイルで False を返すことを確認."""
        assert is_onnx_model(Path("model.pt")) is False


class TestLoadProcessor:
    """_load_processor のテスト."""

    def test_onnx_with_local_processor(self, tmp_path: Path) -> None:
        """同じディレクトリに processor がある場合に読み込めることを確認."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"dummy")

        processor_config = tmp_path / "preprocessor_config.json"
        processor_config.write_text(
            json.dumps(
                {
                    "do_normalize": True,
                    "do_resize": True,
                    "image_processor_type": "RTDetrImageProcessor",
                    "size": {"height": 64, "width": 64},
                }
            )
        )

        processor = _load_processor(onnx_file, {})
        assert processor is not None

    def test_onnx_with_model_name_fallback(self, tmp_path: Path) -> None:
        """Processor がローカルになく model_name からフォールバックすることを確認."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"dummy")

        config = {"model_name": "PekingU/rtdetr_r18vd"}
        processor = _load_processor(onnx_file, config)
        assert processor is not None

    def test_onnx_without_processor_raises_runtime_error(self, tmp_path: Path) -> None:
        """Processor も model_name もない場合 RuntimeError が発生することを確認."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"dummy")

        with pytest.raises(RuntimeError, match="RTDetrImageProcessor を解決できません"):
            _load_processor(onnx_file, {})
