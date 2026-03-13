"""推論バックエンドの検証ユーティリティ."""

from pathlib import Path
from typing import Any


def validate_inputs(
    inputs: dict[str, Any],
    input_names: tuple[str, ...],
    backend_name: str,
) -> None:
    """必須入力キーの存在を検証する.

    Args:
        inputs: 入力テンソルの辞書.
        input_names: 必須入力名のタプル.
        backend_name: エラーメッセージに表示するバックエンド名.

    Raises:
        ValueError: 必須入力キーが不足している場合.
    """
    missing = [name for name in input_names if name not in inputs]
    if missing:
        raise ValueError(
            f"{backend_name}入力が不足しています: {missing}. "
            f"利用可能なキー: {list(inputs.keys())}"
        )


def validate_model_file(
    path: Path,
    label: str,
    expected_suffix: str,
) -> None:
    """モデルファイルの存在・形式を検証する.

    Args:
        path: モデルファイルのパス.
        label: エラーメッセージに表示する名前 (例: "ONNXモデル", "TensorRTエンジン").
        expected_suffix: 期待するファイル拡張子 (例: ".onnx", ".engine").

    Raises:
        FileNotFoundError: ファイルが存在しない場合.
        ValueError: パスがファイルでない, または拡張子が一致しない場合.
    """
    if not path.exists():
        raise FileNotFoundError(f"{label}が見つかりません: {path}")
    if not path.is_file():
        raise ValueError(f"{label}のパスはファイルである必要があります: {path}")
    if path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"{label}のファイル拡張子は {expected_suffix} である必要があります: "
            f"{path}"
        )
