"""scripts/common/inference.py のテスト."""

from pathlib import Path

import pytest

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.scripts.common.inference import (
    collect_image_files,
    resolve_model_path,
    resolve_pipeline_mode,
)


class TestResolveModelPath:
    """resolve_model_path のテスト."""

    def test_returns_path_when_model_dir_exists(self, tmp_path: Path) -> None:
        """model_dir が存在する場合そのパスを返すことを確認."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config: DetectionConfigDict = {"work_dir": str(tmp_path)}

        result = resolve_model_path(config, str(model_dir))

        assert result == model_dir

    def test_returns_none_when_model_dir_not_exists(self, tmp_path: Path) -> None:
        """model_dir が存在しない場合 None を返すことを確認."""
        config: DetectionConfigDict = {"work_dir": str(tmp_path)}

        result = resolve_model_path(config, str(tmp_path / "nonexistent"))

        assert result is None

    def test_returns_pretrained_when_no_workspaces(self, tmp_path: Path) -> None:
        """model_dir=None でワークスペースが無い場合 PRETRAINED を返すことを確認."""
        from pochidetection.scripts.common.inference import PRETRAINED

        work_dir = tmp_path / "work_dirs"
        work_dir.mkdir()
        config: DetectionConfigDict = {"work_dir": str(work_dir)}

        result = resolve_model_path(config, None)

        assert result == PRETRAINED

    def test_returns_best_from_latest_workspace(self, tmp_path: Path) -> None:
        """model_dir=None で最新ワークスペースの best を返すことを確認."""
        work_dir = tmp_path / "work_dirs"
        workspace = work_dir / "20260101_001"
        best_dir = workspace / "best"
        best_dir.mkdir(parents=True)
        config: DetectionConfigDict = {"work_dir": str(work_dir)}

        result = resolve_model_path(config, None)

        assert result == best_dir

    def test_returns_none_when_best_not_exists(self, tmp_path: Path) -> None:
        """model_dir=None で best ディレクトリが無い場合 None を返すことを確認."""
        work_dir = tmp_path / "work_dirs"
        workspace = work_dir / "20260101_001"
        workspace.mkdir(parents=True)
        config: DetectionConfigDict = {"work_dir": str(work_dir)}

        result = resolve_model_path(config, None)

        assert result is None


class TestCollectImageFiles:
    """collect_image_files のテスト."""

    def test_returns_none_when_dir_not_exists(self, tmp_path: Path) -> None:
        """ディレクトリが存在しない場合 None を返すことを確認."""
        result = collect_image_files(str(tmp_path / "nonexistent"))

        assert result is None

    def test_returns_none_when_no_images(self, tmp_path: Path) -> None:
        """画像ファイルが無い場合 None を返すことを確認."""
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b")

        result = collect_image_files(str(tmp_path))

        assert result is None

    def test_returns_image_files(self, tmp_path: Path) -> None:
        """画像ファイルを正しく収集することを確認."""
        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8")
        (tmp_path / "diagram.png").write_bytes(b"\x89PNG")
        (tmp_path / "notes.txt").write_text("not an image")

        result = collect_image_files(str(tmp_path))

        assert result is not None
        assert len(result) == 2
        names = {f.name for f in result}
        assert names == {"photo.jpg", "diagram.png"}

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        """大文字拡張子の画像ファイルも収集することを確認."""
        (tmp_path / "photo.JPG").write_bytes(b"\xff\xd8")
        (tmp_path / "image.PNG").write_bytes(b"\x89PNG")

        result = collect_image_files(str(tmp_path))

        assert result is not None
        assert len(result) == 2


class TestResolvePipelineMode:
    """resolve_pipeline_mode のテスト."""

    def test_pytorch_default_returns_gpu(self, tmp_path: Path) -> None:
        """PyTorch backend (default) は requested=None で 'gpu' を返す."""
        model_path = tmp_path / "model.pt"
        assert resolve_pipeline_mode(None, model_path) == "gpu"

    def test_tensorrt_default_returns_gpu(self, tmp_path: Path) -> None:
        """TensorRT backend (.engine) は requested=None で 'gpu' を返す."""
        model_path = tmp_path / "model.engine"
        assert resolve_pipeline_mode(None, model_path) == "gpu"

    def test_onnx_default_returns_cpu(self, tmp_path: Path) -> None:
        """ONNX backend (.onnx) は requested=None で 'cpu' を返す (自動解決)."""
        model_path = tmp_path / "model.onnx"
        assert resolve_pipeline_mode(None, model_path) == "cpu"

    def test_explicit_cpu_with_pytorch(self, tmp_path: Path) -> None:
        """PyTorch + 明示 'cpu' は 'cpu' を返す."""
        model_path = tmp_path / "model.pt"
        assert resolve_pipeline_mode("cpu", model_path) == "cpu"

    def test_explicit_gpu_with_tensorrt(self, tmp_path: Path) -> None:
        """TensorRT + 明示 'gpu' は 'gpu' を返す."""
        model_path = tmp_path / "model.engine"
        assert resolve_pipeline_mode("gpu", model_path) == "gpu"

    def test_explicit_cpu_with_onnx(self, tmp_path: Path) -> None:
        """ONNX + 明示 'cpu' は 'cpu' を返す."""
        model_path = tmp_path / "model.onnx"
        assert resolve_pipeline_mode("cpu", model_path) == "cpu"

    def test_explicit_gpu_with_onnx_raises_value_error(self, tmp_path: Path) -> None:
        """ONNX + 明示 'gpu' は ValueError で起動拒否."""
        model_path = tmp_path / "model.onnx"
        with pytest.raises(ValueError, match="ONNX backend は --pipeline cpu のみ対応"):
            resolve_pipeline_mode("gpu", model_path)
