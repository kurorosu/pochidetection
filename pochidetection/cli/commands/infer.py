"""infer コマンドの実行ロジック."""

import argparse
import sys
from collections.abc import Callable
from typing import Any

from pochidetection.cli.parser import DEFAULT_CONFIG
from pochidetection.utils import ConfigLoader
from pochidetection.utils.config_resolver import resolve_config_path


def _resolve_infer(
    config: dict[str, Any],
) -> Callable[[dict[str, Any], str, str | None, str | None], None]:
    """Architecture に基づいて infer 関数を返す.

    Args:
        config: 設定辞書.

    Returns:
        infer 関数.
    """
    arch = config.get("architecture", "RTDetr")
    if arch == "SSDLite":
        from pochidetection.scripts.ssdlite.infer import infer as ssdlite_infer

        return ssdlite_infer

    from pochidetection.scripts.rtdetr.infer import infer as rtdetr_infer

    return rtdetr_infer


def run_infer(args: argparse.Namespace) -> None:
    """Infer コマンドを実行する.

    Args:
        args: パース済みの引数.
    """
    config_path = resolve_config_path(args.config, args.model_dir, DEFAULT_CONFIG)
    config = ConfigLoader.load(config_path)
    image_dir = args.dir or config.get("infer_image_dir")
    if image_dir is None:
        print(
            "Error: 推論対象の画像ディレクトリが未指定です. "
            "-d オプションまたは config の infer_image_dir を設定してください.",
            file=sys.stderr,
        )
        sys.exit(1)
    infer_fn = _resolve_infer(config)
    infer_fn(config, image_dir, args.model_dir, config_path)
