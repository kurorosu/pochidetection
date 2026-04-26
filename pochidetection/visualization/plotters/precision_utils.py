"""Precision/scores テンソルの共通ユーティリティ.

torchmetrics MeanAveragePrecision は GT 0 件のクラスや無効ポイントを ``-1`` で表すため,
描画前に NaN へ置換する処理が plotter 横断で必要になる. その置換を 1 箇所に集約する.
"""

import torch

__all__ = ["replace_invalid_with_nan"]


def replace_invalid_with_nan(
    tensor: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """無効値 (``< 0``) を NaN に置換する.

    Args:
        tensor: 置換対象のテンソル. dtype は float 系を想定.
        valid_mask: 有効ポイントの真偽値マスク. ``None`` の場合は ``tensor >= 0``
            を使う. F1 plotter のように precision の mask を scores にも同じ位置で
            適用したい場合に明示的に渡す.

    Returns:
        無効位置を NaN に置き換えた新規テンソル.
    """
    if valid_mask is None:
        valid_mask = tensor >= 0
    return torch.where(valid_mask, tensor, torch.full_like(tensor, float("nan")))
