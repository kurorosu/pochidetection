"""train コマンドの実行ロジック."""

import argparse

from pochidetection.cli.registry import resolve_train
from pochidetection.utils import ConfigLoader


def run_train(args: argparse.Namespace) -> None:
    """Train コマンドを実行する.

    Args:
        args: パース済みの引数.
    """
    config = ConfigLoader.load(args.config)
    train_fn = resolve_train(config)
    train_fn(config, args.config)
