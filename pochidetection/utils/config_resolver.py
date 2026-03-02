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
    2. config 未指定かつ model_dir 指定あり → モデルパスの親から config.py を探す
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
        if model_path.suffix.lower() in (".onnx", ".engine"):
            candidate = model_path.parent / "config.py"
        else:
            candidate = model_path.parent / "config.py"

        if candidate.exists():
            logger.info(f"Auto-resolved config: {candidate}")
            return str(candidate)

        logger.warning(
            f"Config not found at {candidate}, falling back to {default_config}"
        )

    return default_config
