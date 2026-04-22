"""pipelines/model_path.py のテスト."""

from pathlib import Path

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.pipelines.model_path import PRETRAINED, _resolve_model_path


class TestResolveModelPath:
    """_resolve_model_path のテスト."""

    def test_returns_path_when_model_dir_exists(self, tmp_path: Path) -> None:
        """model_dir が存在する場合そのパスを返すことを確認."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config: DetectionConfigDict = {"work_dir": str(tmp_path)}

        result = _resolve_model_path(config, str(model_dir))

        assert result == model_dir

    def test_returns_none_when_model_dir_not_exists(self, tmp_path: Path) -> None:
        """model_dir が存在しない場合 None を返すことを確認."""
        config: DetectionConfigDict = {"work_dir": str(tmp_path)}

        result = _resolve_model_path(config, str(tmp_path / "nonexistent"))

        assert result is None

    def test_returns_pretrained_when_no_workspaces(self, tmp_path: Path) -> None:
        """model_dir=None でワークスペースが無い場合 PRETRAINED を返すことを確認."""
        work_dir = tmp_path / "work_dirs"
        work_dir.mkdir()
        config: DetectionConfigDict = {"work_dir": str(work_dir)}

        result = _resolve_model_path(config, None)

        assert result == PRETRAINED

    def test_returns_best_from_latest_workspace(self, tmp_path: Path) -> None:
        """model_dir=None で最新ワークスペースの best を返すことを確認."""
        work_dir = tmp_path / "work_dirs"
        workspace = work_dir / "20260101_001"
        best_dir = workspace / "best"
        best_dir.mkdir(parents=True)
        config: DetectionConfigDict = {"work_dir": str(work_dir)}

        result = _resolve_model_path(config, None)

        assert result == best_dir

    def test_returns_none_when_best_not_exists(self, tmp_path: Path) -> None:
        """model_dir=None で best ディレクトリが無い場合 None を返すことを確認."""
        work_dir = tmp_path / "work_dirs"
        workspace = work_dir / "20260101_001"
        workspace.mkdir(parents=True)
        config: DetectionConfigDict = {"work_dir": str(work_dir)}

        result = _resolve_model_path(config, None)

        assert result is None
