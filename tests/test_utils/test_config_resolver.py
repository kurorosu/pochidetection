"""resolve_config_path のテスト."""

from pathlib import Path

from pochidetection.utils.config_resolver import resolve_config_path

DEFAULT_CONFIG = "configs/rtdetr_coco.py"


class TestResolveConfigPath:
    """resolve_config_path のテスト."""

    def test_explicit_config_takes_precedence(self) -> None:
        """明示的な -c 指定が優先されることを確認."""
        result = resolve_config_path("custom.py", "work_dirs/xxx/best", DEFAULT_CONFIG)
        assert result == "custom.py"

    def test_auto_resolve_workspace_best(self, tmp_path: Path) -> None:
        """workspace/best 指定時に親の config.py を自動検出することを確認."""
        workspace = tmp_path / "work_dirs" / "20260228_001"
        best_dir = workspace / "best"
        best_dir.mkdir(parents=True)
        config_file = workspace / "config.py"
        config_file.write_text('data_root = "data"\n', encoding="utf-8")

        result = resolve_config_path(None, str(best_dir), DEFAULT_CONFIG)
        assert result == str(config_file)

    def test_auto_resolve_workspace_last(self, tmp_path: Path) -> None:
        """workspace/last 指定時に親の config.py を自動検出することを確認."""
        workspace = tmp_path / "work_dirs" / "20260228_001"
        last_dir = workspace / "last"
        last_dir.mkdir(parents=True)
        config_file = workspace / "config.py"
        config_file.write_text('data_root = "data"\n', encoding="utf-8")

        result = resolve_config_path(None, str(last_dir), DEFAULT_CONFIG)
        assert result == str(config_file)

    def test_auto_resolve_onnx_model(self, tmp_path: Path) -> None:
        """ONNX ファイル指定時に同ディレクトリの config.py を自動検出することを確認."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        onnx_file = model_dir / "model.onnx"
        onnx_file.touch()
        config_file = model_dir / "config.py"
        config_file.write_text('data_root = "data"\n', encoding="utf-8")

        result = resolve_config_path(None, str(onnx_file), DEFAULT_CONFIG)
        assert result == str(config_file)

    def test_auto_resolve_engine_model(self, tmp_path: Path) -> None:
        """TensorRT エンジン指定時に同ディレクトリの config.py を自動検出することを確認."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        engine_file = model_dir / "model.engine"
        engine_file.touch()
        config_file = model_dir / "config.py"
        config_file.write_text('data_root = "data"\n', encoding="utf-8")

        result = resolve_config_path(None, str(engine_file), DEFAULT_CONFIG)
        assert result == str(config_file)

    def test_fallback_when_config_not_found(self, tmp_path: Path) -> None:
        """workspace に config.py がない場合にデフォルトにフォールバックすることを確認."""
        best_dir = tmp_path / "work_dirs" / "20260228_001" / "best"
        best_dir.mkdir(parents=True)

        result = resolve_config_path(None, str(best_dir), DEFAULT_CONFIG)
        assert result == DEFAULT_CONFIG

    def test_fallback_when_no_model_dir(self) -> None:
        """model_dir 未指定時にデフォルトを返すことを確認."""
        result = resolve_config_path(None, None, DEFAULT_CONFIG)
        assert result == DEFAULT_CONFIG
