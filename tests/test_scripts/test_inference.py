"""scripts/common/inference.py のテスト."""

from pathlib import Path
from typing import Any

from pochidetection.scripts.common.inference import (
    collect_image_files,
    resolve_model_path,
)


class TestResolveModelPath:
    """resolve_model_path のテスト."""

    def test_returns_path_when_model_dir_exists(self, tmp_path: Path) -> None:
        """model_dir が存在する場合そのパスを返すことを確認."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config: dict[str, Any] = {"work_dir": str(tmp_path)}

        result = resolve_model_path(config, str(model_dir))

        assert result == model_dir

    def test_returns_none_when_model_dir_not_exists(self, tmp_path: Path) -> None:
        """model_dir が存在しない場合 None を返すことを確認."""
        config: dict[str, Any] = {"work_dir": str(tmp_path)}

        result = resolve_model_path(config, str(tmp_path / "nonexistent"))

        assert result is None

    def test_returns_none_when_no_workspaces(self, tmp_path: Path) -> None:
        """model_dir=None でワークスペースが無い場合 None を返すことを確認."""
        work_dir = tmp_path / "work_dirs"
        work_dir.mkdir()
        config: dict[str, Any] = {"work_dir": str(work_dir)}

        result = resolve_model_path(config, None)

        assert result is None

    def test_returns_best_from_latest_workspace(self, tmp_path: Path) -> None:
        """model_dir=None で最新ワークスペースの best を返すことを確認."""
        work_dir = tmp_path / "work_dirs"
        workspace = work_dir / "20260101_001"
        best_dir = workspace / "best"
        best_dir.mkdir(parents=True)
        config: dict[str, Any] = {"work_dir": str(work_dir)}

        result = resolve_model_path(config, None)

        assert result == best_dir

    def test_returns_none_when_best_not_exists(self, tmp_path: Path) -> None:
        """model_dir=None で best ディレクトリが無い場合 None を返すことを確認."""
        work_dir = tmp_path / "work_dirs"
        workspace = work_dir / "20260101_001"
        workspace.mkdir(parents=True)
        config: dict[str, Any] = {"work_dir": str(work_dir)}

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
