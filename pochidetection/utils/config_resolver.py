"""設定ファイルパスの自動解決."""

from pathlib import Path

from pochidetection.logging import LoggerManager

logger = LoggerManager().get_logger(__name__)


def resolve_config_path(
    config: str | None,
    model_dir: str | None,
    default_config: str,
) -> str:
    """推論時の設定ファイルパスを解決する.

    優先順位:
    1. config 指定あり → そのパスを使う
    2. config 未指定かつ model_dir 指定あり → モデルパスから .py ファイルを探す
    3. config 未指定かつ model_dir 未指定 → デフォルト config を使う

    Args:
        config: 明示的に指定された設定ファイルパス. None の場合は自動解決を試みる.
        model_dir: モデルディレクトリまたはモデルファイルのパス.
        default_config: フォールバック用のデフォルト設定ファイルパス.

    Returns:
        設定ファイルのパス.
    """
    if config is not None:
        logger.info(f"Using explicit config: {config}")
        return config

    if model_dir is not None:
        model_path = Path(model_dir)

        # モデルファイル (.onnx, .engine) → 同ディレクトリ, その親
        # ディレクトリ (best, last 等) → 親ディレクトリ, 自身
        if model_path.suffix.lower() in (".onnx", ".engine"):
            base_dir = model_path.parent
        else:
            base_dir = model_path

        search_dirs = [base_dir, base_dir.parent]

        for search_dir in search_dirs:
            found = _find_config_in_dir(search_dir)
            if found is not None:
                logger.info(f"Auto-resolved config: {found}")
                return str(found)

        searched = ", ".join(str(d) for d in search_dirs)
        logger.warning(
            f"Config not found in [{searched}], falling back to {default_config}"
        )

    return default_config


def _find_config_in_dir(directory: Path) -> Path | None:
    """ディレクトリから設定ファイルを探す.

    config.py を優先し, なければ唯一の .py ファイルを返す.

    Args:
        directory: 検索対象のディレクトリ.

    Returns:
        見つかった設定ファイルのパス. 見つからなければ None.
    """
    config_py = directory / "config.py"
    if config_py.exists():
        return config_py

    py_files = list(directory.glob("*.py"))
    if len(py_files) == 1:
        return py_files[0]

    return None
