"""Learning Rate Scheduler のファクトリ関数."""

from typing import Any

import torch.optim.lr_scheduler as lr_module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def _get_scheduler_class(name: str) -> type[LRScheduler]:
    """torch.optim.lr_scheduler から指定名のクラスを取得.

    Args:
        name: Scheduler クラス名 (例: "CosineAnnealingLR").

    Returns:
        Scheduler クラス.

    Raises:
        ValueError: 存在しない Scheduler 名が指定された場合.
    """
    if not hasattr(lr_module, name):
        raise ValueError(
            f"不明な lr_scheduler: '{name}'. "
            f"torch.optim.lr_scheduler に存在するクラス名を指定してください."
        )
    cls = getattr(lr_module, name)
    if not (isinstance(cls, type) and issubclass(cls, LRScheduler)):
        raise ValueError(f"'{name}' は LRScheduler のサブクラスではありません.")
    return cls


def build_scheduler(
    optimizer: Optimizer,
    scheduler_name: str | None,
    scheduler_params: dict[str, Any] | None,
    epochs: int,
) -> LRScheduler | None:
    """Config 設定から LR Scheduler を生成.

    Args:
        optimizer: PyTorch Optimizer.
        scheduler_name: Scheduler クラス名. None で無効.
        scheduler_params: Scheduler コンストラクタに渡す追加パラメータ.
        epochs: 学習エポック数 (CosineAnnealingLR の T_max 等に使用).

    Returns:
        LRScheduler インスタンス, または None (無効時).
    """
    if scheduler_name is None:
        return None

    scheduler_cls = _get_scheduler_class(scheduler_name)
    params = dict(scheduler_params) if scheduler_params else {}

    # CosineAnnealingLR: T_max が未指定の場合はエポック数をデフォルトとする
    if scheduler_name == "CosineAnnealingLR" and "T_max" not in params:
        params["T_max"] = epochs

    return scheduler_cls(optimizer, **params)
