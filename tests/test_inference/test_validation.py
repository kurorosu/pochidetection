"""推論バックエンドの検証ユーティリティのテスト."""

from pathlib import Path

import pytest

from pochidetection.inference.validation import validate_inputs, validate_model_file


class TestValidateInputs:
    """validate_inputs のテスト."""

    def test_all_keys_present_passes(self) -> None:
        """全ての必須キーが揃っている場合, 例外が発生しない."""
        inputs = {"pixel_values": [1, 2, 3], "attention_mask": [1, 1, 1]}
        validate_inputs(inputs, ("pixel_values", "attention_mask"), "TestBackend")

    def test_missing_key_raises_value_error(self) -> None:
        """必須キーが不足している場合, ValueError が発生する."""
        inputs = {"pixel_values": [1, 2, 3]}
        with pytest.raises(ValueError, match="attention_mask"):
            validate_inputs(inputs, ("pixel_values", "attention_mask"), "TestBackend")

    def test_error_message_includes_backend_name(self) -> None:
        """エラーメッセージにバックエンド名が含まれる."""
        with pytest.raises(ValueError, match="MyBackend"):
            validate_inputs({}, ("key1",), "MyBackend")

    def test_error_message_includes_available_keys(self) -> None:
        """エラーメッセージに利用可能なキーが含まれる."""
        inputs = {"existing_key": 1}
        with pytest.raises(ValueError, match="existing_key"):
            validate_inputs(inputs, ("missing_key",), "Backend")

    def test_empty_input_names_passes(self) -> None:
        """必須キーが空の場合, 常に通過する."""
        validate_inputs({}, (), "Backend")

    def test_multiple_missing_keys(self) -> None:
        """複数のキーが不足している場合, 全て報告される."""
        with pytest.raises(ValueError, match="key1") as exc_info:
            validate_inputs({}, ("key1", "key2"), "Backend")
        assert "key2" in str(exc_info.value)


class TestValidateModelFile:
    """validate_model_file のテスト."""

    def test_valid_onnx_file_passes(self, tmp_path: Path) -> None:
        """有効な ONNX ファイルで例外が発生しない."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"dummy")
        validate_model_file(model_file, "ONNXモデル", ".onnx")

    def test_nonexistent_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """存在しないファイルで FileNotFoundError が発生する."""
        model_file = tmp_path / "missing.onnx"
        with pytest.raises(FileNotFoundError, match="見つかりません"):
            validate_model_file(model_file, "ONNXモデル", ".onnx")

    def test_directory_raises_value_error(self, tmp_path: Path) -> None:
        """パスがディレクトリの場合, ValueError が発生する."""
        dir_path = tmp_path / "model.onnx"
        dir_path.mkdir()
        with pytest.raises(ValueError, match="ファイルである必要"):
            validate_model_file(dir_path, "ONNXモデル", ".onnx")

    def test_wrong_suffix_raises_value_error(self, tmp_path: Path) -> None:
        """拡張子が一致しない場合, ValueError が発生する."""
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"dummy")
        with pytest.raises(ValueError, match=r"\.onnx"):
            validate_model_file(model_file, "ONNXモデル", ".onnx")

    def test_case_insensitive_suffix(self, tmp_path: Path) -> None:
        """大文字拡張子でも通過する."""
        model_file = tmp_path / "model.ONNX"
        model_file.write_bytes(b"dummy")
        validate_model_file(model_file, "ONNXモデル", ".onnx")

    def test_engine_suffix(self, tmp_path: Path) -> None:
        """TensorRT エンジンファイルの検証が通る."""
        engine_file = tmp_path / "model.engine"
        engine_file.write_bytes(b"dummy")
        validate_model_file(engine_file, "TensorRTエンジン", ".engine")
