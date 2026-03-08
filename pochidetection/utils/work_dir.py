"""作業ディレクトリ管理ユーティリティ.

yyyymmdd_001形式の作業ディレクトリ自動生成機能を提供する.
"""

import re
import shutil
from datetime import datetime
from pathlib import Path


def get_current_date_str() -> str:
    """現在の日付をyyyymmdd形式で取得.

    Returns:
        現在の日付文字列 (例: "20260124").
    """
    return datetime.now().strftime("%Y%m%d")


def find_next_index(base_dir: Path, date_str: str) -> int:
    """同じ日付で既存のディレクトリがある場合, 次のインデックスを取得.

    Args:
        base_dir: 検索対象のベースディレクトリ.
        date_str: 日付文字列 (例: "20260124").

    Returns:
        次のインデックス (1, 2, 3, ...).

    Examples:
        work_dirs/20260124_001/ が存在 → 2 を返す
        work_dirs/20260124_002/ も存在 → 3 を返す
        該当なし → 1 を返す
    """
    if not base_dir.exists():
        return 1

    pattern = re.compile(rf"^{date_str}_(\d{{3}})$")
    indices = []

    for item in base_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                indices.append(int(match.group(1)))

    if not indices:
        return 1

    return max(indices) + 1


def format_workspace_name(date_str: str, index: int) -> str:
    """日付とインデックスからワークスペース名を生成.

    Args:
        date_str: 日付文字列 (例: "20260124").
        index: インデックス.

    Returns:
        ワークスペース名 (例: "20260124_001").
    """
    return f"{date_str}_{index:03d}"


def parse_workspace_name(dirname: str) -> tuple[str, int]:
    """ディレクトリ名から日付とインデックスを分離.

    Args:
        dirname: ディレクトリ名 (例: "20260124_001").

    Returns:
        (日付文字列, インデックス) のタプル.

    Raises:
        ValueError: 形式が正しくない場合.

    Examples:
        "20260124_001" → ("20260124", 1)
        "20260124_123" → ("20260124", 123)
    """
    pattern = re.compile(r"^(\d{8})_(\d{3})$")
    match = pattern.match(dirname)

    if not match:
        raise ValueError(f"Invalid directory name format: {dirname}")

    date_str = match.group(1)
    index = int(match.group(2))

    return date_str, index


class WorkspaceManager:
    """作業ディレクトリ管理クラス.

    yyyymmdd_xxx 形式のディレクトリ構造を管理し,
    モデル保存, 設定ファイル, 学習履歴の保存先を提供する.

    Attributes:
        base_dir: ベースディレクトリのパス.
        current_workspace: 現在のワークスペースのパス.
    """

    def __init__(self, base_dir: str | Path = "work_dirs") -> None:
        """WorkspaceManagerを初期化.

        Args:
            base_dir: ベースディレクトリのパス.
        """
        self._base_dir = Path(base_dir)
        self._current_workspace: Path | None = None

    @property
    def base_dir(self) -> Path:
        """ベースディレクトリを取得.

        Returns:
            ベースディレクトリのパス.
        """
        return self._base_dir

    @property
    def current_workspace(self) -> Path | None:
        """現在のワークスペースを取得.

        Returns:
            現在のワークスペースのパス. 未作成の場合はNone.
        """
        return self._current_workspace

    def _ensure_workspace_created(self) -> Path:
        """ワークスペースが作成済みであることを確認する.

        Returns:
            現在のワークスペースのパス.

        Raises:
            RuntimeError: ワークスペースが作成されていない場合.
        """
        if self._current_workspace is None:
            raise RuntimeError(
                "ワークスペースが作成されていません. "
                "create_workspace() を先に呼び出してください."
            )
        return self._current_workspace

    def create_workspace(self) -> Path:
        """新しいワークスペースを作成.

        yyyymmdd_xxx 形式のディレクトリを作成し,
        必要なサブディレクトリも作成する.

        Returns:
            作成されたワークスペースのパス.

        Examples:
            work_dirs/20260124_001/
            work_dirs/20260124_001/best/
            work_dirs/20260124_001/last/
        """
        self._base_dir.mkdir(parents=True, exist_ok=True)

        date_str = get_current_date_str()
        next_index = find_next_index(self._base_dir, date_str)
        workspace_name = format_workspace_name(date_str, next_index)
        workspace_path = self._base_dir / workspace_name

        workspace_path.mkdir(parents=True, exist_ok=True)

        self._current_workspace = workspace_path

        return workspace_path

    def get_best_dir(self) -> Path:
        """ベストモデル保存用ディレクトリのパスを取得.

        Returns:
            ベストモデル保存用ディレクトリのパス.

        Raises:
            RuntimeError: ワークスペースが作成されていない場合.
        """
        return self._ensure_workspace_created() / "best"

    def get_last_dir(self) -> Path:
        """最終モデル保存用ディレクトリのパスを取得.

        Returns:
            最終モデル保存用ディレクトリのパス.

        Raises:
            RuntimeError: ワークスペースが作成されていない場合.
        """
        return self._ensure_workspace_created() / "last"

    def save_config(self, config_path: str | Path) -> Path:
        """設定ファイルをワークスペースにコピー.

        元のファイル名を保持してコピーする.

        Args:
            config_path: コピー元の設定ファイルパス.

        Returns:
            コピー先のファイルパス.

        Raises:
            RuntimeError: ワークスペースが作成されていない場合.
            FileNotFoundError: コピー元ファイルが存在しない場合.
        """
        workspace = self._ensure_workspace_created()

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

        target_path = workspace / config_path.name
        shutil.copy2(config_path, target_path)

        return target_path

    def get_available_workspaces(self) -> list[dict[str, str | int | bool]]:
        """利用可能なワークスペースの一覧を取得.

        Returns:
            ワークスペース情報のリスト.
        """
        if not self._base_dir.exists():
            return []

        workspaces: list[dict[str, str | int | bool]] = []

        for item in self._base_dir.iterdir():
            if item.is_dir():
                try:
                    date_str, index = parse_workspace_name(item.name)
                    workspaces.append(
                        {
                            "name": item.name,
                            "path": str(item),
                            "date": date_str,
                            "index": index,
                            "exists": item.exists(),
                        }
                    )
                except ValueError:
                    continue

        workspaces.sort(key=lambda x: (str(x["date"]), int(str(x["index"]))))

        return workspaces
