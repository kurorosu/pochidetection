"""train コマンドの実行ロジック."""

import argparse
from collections.abc import Callable
from typing import Any

from pochidetection.utils import ConfigLoader


def _resolve_train(
    config: dict[str, Any],
) -> Callable[[dict[str, Any], str], None]:
    """Architecture に基づいて train 関数を返す.

    Args:
        config: 設定辞書.

    Returns:
        train 関数.
    """
    arch = config.get("architecture", "RTDetr")
    if arch == "SSDLite":
        # 未使用モデルの import コストを避けるため lazy import
        from pochidetection.scripts.ssdlite.train import train as ssdlite_train

        return ssdlite_train

    from pochidetection.scripts.rtdetr.train import train as rtdetr_train

    return rtdetr_train


def run_train(args: argparse.Namespace) -> None:
    """Train コマンドを実行する.

    Args:
        args: パース済みの引数.
    """
    config = ConfigLoader.load(args.config)
    train_fn = _resolve_train(config)
    train_fn(config, args.config)
