"""推論バックエンドの入力検証ユーティリティ."""

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
