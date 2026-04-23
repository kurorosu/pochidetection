"""モデルパス判定 / 解決のユーティリティ.

推論 pipeline 構築の最前段で使う, モデルパス関連の純粋ロジック:

- ``PRETRAINED``: pretrained モデル利用を示すセンチネル Path.
- ``is_onnx_model`` / ``is_tensorrt_model``: モデルファイル種別判定.
- ``resolve_model_path``: CLI / config から推論対象のモデルパスを解決.
"""

from pathlib import Path

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.logging import LoggerManager
from pochidetection.utils import WorkspaceManager

__all__ = [
    "PRETRAINED",
    "is_onnx_model",
    "is_tensorrt_model",
    "resolve_model_path",
]

logger = LoggerManager().get_logger(__name__)

PRETRAINED = Path("__pretrained__")
"""プリトレインモデル使用を示すセンチネル値."""


def is_onnx_model(model_path: Path) -> bool:
    """モデルパスが ONNX ファイルかどうかを判定する.

    Args:
        model_path: モデルのパス.

    Returns:
        .onnx ファイルの場合 True.
    """
    return model_path.suffix.lower() == ".onnx"


def is_tensorrt_model(model_path: Path) -> bool:
    """モデルパスが TensorRT エンジンかどうかを判定する.

    Args:
        model_path: モデルのパス.

    Returns:
        .engine ファイルの場合 True.
    """
    return model_path.suffix.lower() == ".engine"


def resolve_model_path(
    config: DetectionConfigDict,
    model_dir: str | None,
) -> Path | None:
    """モデルパスを解決.

    Args:
        config: 設定辞書.
        model_dir: 指定されたモデルディレクトリ.

    Returns:
        モデルパス. エラー時は None.
    """
    if model_dir is not None:
        model_path = Path(model_dir)
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return None
        return model_path

    workspace_manager = WorkspaceManager(config["work_dir"])
    workspaces = workspace_manager.get_available_workspaces()

    if not workspaces:
        logger.info(
            "No trained models found. Using COCO pretrained model for inference."
        )
        return PRETRAINED

    latest_workspace = Path(str(workspaces[-1]["path"]))
    model_path = latest_workspace / "best"

    if not model_path.exists():
        logger.error(
            f"Best model not found at {model_path}. Please run training first."
        )
        return None

    return model_path
