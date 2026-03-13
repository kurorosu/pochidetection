"""WorkspaceManagerのテスト."""

import re
from pathlib import Path

import pytest

from pochidetection.utils.work_dir import (
    WorkspaceManager,
    find_next_index,
    format_workspace_name,
    get_current_date_str,
    parse_workspace_name,
)


class TestGetCurrentDateStr:
    """get_current_date_strのテスト."""

    def test_returns_yyyymmdd_format(self) -> None:
        """yyyymmdd形式の文字列を返す."""
        result = get_current_date_str()

        assert len(result) == 8
        assert result.isdigit()
        # 妥当な日付範囲かチェック (2020-2099年)
        year = int(result[:4])
        month = int(result[4:6])
        day = int(result[6:8])
        assert 2020 <= year <= 2099
        assert 1 <= month <= 12
        assert 1 <= day <= 31


class TestFindNextIndex:
    """find_next_indexのテスト."""

    def test_returns_1_when_base_dir_not_exists(self, tmp_path: Path) -> None:
        """ベースディレクトリが存在しない場合は1を返す."""
        non_existent = tmp_path / "non_existent"

        result = find_next_index(non_existent, "20260124")

        assert result == 1

    def test_returns_1_when_no_matching_dirs(self, tmp_path: Path) -> None:
        """マッチするディレクトリがない場合は1を返す."""
        (tmp_path / "other_dir").mkdir()

        result = find_next_index(tmp_path, "20260124")

        assert result == 1

    def test_returns_2_when_001_exists(self, tmp_path: Path) -> None:
        """001が存在する場合は2を返す."""
        (tmp_path / "20260124_001").mkdir()

        result = find_next_index(tmp_path, "20260124")

        assert result == 2

    def test_returns_4_when_001_002_003_exist(self, tmp_path: Path) -> None:
        """001, 002, 003が存在する場合は4を返す."""
        (tmp_path / "20260124_001").mkdir()
        (tmp_path / "20260124_002").mkdir()
        (tmp_path / "20260124_003").mkdir()

        result = find_next_index(tmp_path, "20260124")

        assert result == 4

    def test_ignores_different_date(self, tmp_path: Path) -> None:
        """異なる日付のディレクトリは無視する."""
        (tmp_path / "20260123_001").mkdir()
        (tmp_path / "20260123_002").mkdir()

        result = find_next_index(tmp_path, "20260124")

        assert result == 1


class TestFormatWorkspaceName:
    """format_workspace_nameのテスト."""

    def test_formats_correctly(self) -> None:
        """正しくフォーマットする."""
        result = format_workspace_name("20260124", 1)

        assert result == "20260124_001"

    def test_formats_with_padding(self) -> None:
        """3桁でゼロパディングする."""
        result = format_workspace_name("20260124", 99)

        assert result == "20260124_099"


class TestParseWorkspaceName:
    """parse_workspace_nameのテスト."""

    def test_parses_correctly(self) -> None:
        """正しくパースする."""
        date_str, index = parse_workspace_name("20260124_001")

        assert date_str == "20260124"
        assert index == 1

    def test_parses_large_index(self) -> None:
        """大きいインデックスをパースする."""
        date_str, index = parse_workspace_name("20260124_123")

        assert date_str == "20260124"
        assert index == 123

    def test_raises_on_invalid_format(self) -> None:
        """不正な形式でValueErrorを発生させる."""
        with pytest.raises(ValueError, match="Invalid directory name format"):
            parse_workspace_name("invalid")

    def test_raises_on_wrong_date_length(self) -> None:
        """日付の桁数が違う場合にValueErrorを発生させる."""
        with pytest.raises(ValueError):
            parse_workspace_name("2026012_001")


class TestWorkspaceManager:
    """WorkspaceManagerのテスト."""

    def test_init_sets_base_dir(self, tmp_path: Path) -> None:
        """初期化でベースディレクトリを設定する."""
        manager = WorkspaceManager(tmp_path)

        assert manager.base_dir == tmp_path
        assert manager.current_workspace is None

    def test_create_workspace_creates_directory(self, tmp_path: Path) -> None:
        """create_workspaceがディレクトリを作成する."""
        manager = WorkspaceManager(tmp_path)

        workspace = manager.create_workspace()

        assert workspace.exists()
        # yyyymmdd_001 形式であることを確認
        assert re.match(r"^\d{8}_001$", workspace.name)
        assert manager.current_workspace == workspace

    def test_create_workspace_increments_index(self, tmp_path: Path) -> None:
        """create_workspaceがインデックスをインクリメントする."""
        manager = WorkspaceManager(tmp_path)

        # 1回目
        workspace1 = manager.create_workspace()
        # 2回目
        workspace2 = manager.create_workspace()

        # 同じ日付で連番になっていることを確認
        date1 = workspace1.name[:8]
        date2 = workspace2.name[:8]
        index1 = int(workspace1.name[9:])
        index2 = int(workspace2.name[9:])

        assert date1 == date2
        assert index2 == index1 + 1

    def test_get_best_dir_returns_path(self, tmp_path: Path) -> None:
        """get_best_dirがパスを返す."""
        manager = WorkspaceManager(tmp_path)
        manager.create_workspace()

        best_dir = manager.get_best_dir()

        assert best_dir.name == "best"
        assert best_dir.parent == manager.current_workspace

    def test_get_best_dir_raises_without_workspace(self, tmp_path: Path) -> None:
        """ワークスペース未作成でget_best_dirがRuntimeErrorを発生させる."""
        manager = WorkspaceManager(tmp_path)

        with pytest.raises(RuntimeError, match="ワークスペースが作成されていません"):
            manager.get_best_dir()

    def test_get_last_dir_returns_path(self, tmp_path: Path) -> None:
        """get_last_dirがパスを返す."""
        manager = WorkspaceManager(tmp_path)
        manager.create_workspace()

        last_dir = manager.get_last_dir()

        assert last_dir.name == "last"
        assert last_dir.parent == manager.current_workspace

    def test_get_last_dir_raises_without_workspace(self, tmp_path: Path) -> None:
        """ワークスペース未作成でget_last_dirがRuntimeErrorを発生させる."""
        manager = WorkspaceManager(tmp_path)

        with pytest.raises(RuntimeError, match="ワークスペースが作成されていません"):
            manager.get_last_dir()

    def test_save_config_writes_merged_dict(self, tmp_path: Path) -> None:
        """save_configがマージ済み設定辞書をPythonファイルとして保存する."""
        config = {"batch_size": 32, "learning_rate": 0.001}
        work_dir = tmp_path / "work_dirs"
        manager = WorkspaceManager(work_dir)
        manager.create_workspace()

        target = manager.save_config(config, "ssdlite_coco.py")

        assert target.exists()
        assert target.name == "ssdlite_coco.py"
        content = target.read_text()
        assert "batch_size = 32" in content
        assert "learning_rate = 0.001" in content

    def test_save_config_raises_without_workspace(self, tmp_path: Path) -> None:
        """ワークスペース未作成でsave_configがRuntimeErrorを発生させる."""
        manager = WorkspaceManager(tmp_path)

        with pytest.raises(RuntimeError, match="ワークスペースが作成されていません"):
            manager.save_config({"key": "value"}, "config.py")

    def test_get_available_workspaces_returns_empty_when_no_dir(
        self, tmp_path: Path
    ) -> None:
        """ディレクトリがない場合に空リストを返す."""
        manager = WorkspaceManager(tmp_path / "non_existent")

        workspaces = manager.get_available_workspaces()

        assert workspaces == []

    def test_get_available_workspaces_returns_sorted_list(self, tmp_path: Path) -> None:
        """ソートされたワークスペースリストを返す."""
        (tmp_path / "20260124_002").mkdir()
        (tmp_path / "20260124_001").mkdir()
        (tmp_path / "20260123_001").mkdir()
        manager = WorkspaceManager(tmp_path)

        workspaces = manager.get_available_workspaces()

        assert len(workspaces) == 3
        assert workspaces[0]["name"] == "20260123_001"
        assert workspaces[1]["name"] == "20260124_001"
        assert workspaces[2]["name"] == "20260124_002"

    def test_get_available_workspaces_ignores_invalid_dirs(
        self, tmp_path: Path
    ) -> None:
        """不正なディレクトリを無視する."""
        (tmp_path / "20260124_001").mkdir()
        (tmp_path / "invalid_dir").mkdir()
        manager = WorkspaceManager(tmp_path)

        workspaces = manager.get_available_workspaces()

        assert len(workspaces) == 1
        assert workspaces[0]["name"] == "20260124_001"
