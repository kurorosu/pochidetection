"""物体検出 CLI エントリーポイント.

RT-DETR / SSDLite モデルの学習・推論・エクスポートを行うコマンドラインインターフェース.

使用方法:
    uv run pochi train
    uv run pochi train -c configs/rtdetr_coco.py
    uv run pochi train -c configs/ssdlite_coco.py
    uv run pochi infer
    uv run pochi infer -m work_dirs/20260124_001/best
    uv run pochi export -m work_dirs/20260124_001/best
    uv run pochi export -m work_dirs/20260124_001/best/model_fp32.onnx
"""

from pochidetection.cli.commands.export import run_export
from pochidetection.cli.commands.infer import run_infer
from pochidetection.cli.commands.train import run_train
from pochidetection.cli.parser import parse_args
from pochidetection.logging import LoggerManager, LogLevel


def setup_logging(debug: bool = False) -> None:
    """ログ設定の初期化.

    Args:
        debug: DEBUGモードを有効にするか.
    """
    logger_manager = LoggerManager()
    level = LogLevel.DEBUG if debug else LogLevel.INFO
    logger_manager.set_default_level(level)


def main() -> None:
    """メインエントリーポイント."""
    args = parse_args()
    setup_logging(debug=args.debug)

    commands = {
        "train": run_train,
        "infer": run_infer,
        "export": run_export,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
