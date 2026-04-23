"""train コマンドの実行ロジック."""

import argparse

from pochidetection.cli.registry import get_train_for_arch
from pochidetection.utils import ConfigLoader


def run_train(args: argparse.Namespace) -> None:
    """Train コマンドを実行する.

    Args:
        args: パース済みの引数.
    """
    config = ConfigLoader.load(args.config)
    train = get_train_for_arch(config)
    train(config, args.config)
