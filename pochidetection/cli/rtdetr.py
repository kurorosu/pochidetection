"""RT-DETR CLI.

RT-DETRモデルの学習・推論を行うコマンドラインインターフェース.

使用方法:
    uv run pochidet-rtdetr train
    uv run pochidet-rtdetr train -c configs/rtdetr_coco.py
    uv run pochidet-rtdetr infer -i image.jpg
    uv run pochidet-rtdetr infer -i image.jpg -t 0.3 -m work_dirs/20260124_001/best
"""

import argparse

from pochidetection.scripts.rtdetr.infer import infer
from pochidetection.scripts.rtdetr.train import train
from pochidetection.utils import ConfigLoader

DEFAULT_CONFIG = "configs/rtdetr_coco.py"


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース.

    Returns:
        パースされた引数.
    """
    parser = argparse.ArgumentParser(
        description="RT-DETR ファインチューニング・推論CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  学習:
    uv run pochidet-rtdetr train
    uv run pochidet-rtdetr train -c configs/rtdetr_coco.py

  推論:
    uv run pochidet-rtdetr infer -i image.jpg
    uv run pochidet-rtdetr infer -i image.jpg -t 0.3
    uv run pochidet-rtdetr infer -i image.jpg -m work_dirs/20260124_001/best
        """,
    )

    # サブコマンド
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")

    # 学習コマンド
    train_parser = subparsers.add_parser("train", help="モデルの学習")
    train_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"設定ファイルのパス (default: {DEFAULT_CONFIG})",
    )

    # 推論コマンド
    infer_parser = subparsers.add_parser("infer", help="画像の推論")
    infer_parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="推論対象の画像パス",
    )
    infer_parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="検出信頼度閾値 (default: 0.5)",
    )
    infer_parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        default=None,
        help="モデルディレクトリ (default: 最新ワークスペースのbest)",
    )
    infer_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"設定ファイルのパス (default: {DEFAULT_CONFIG})",
    )

    return parser.parse_args()


def main() -> None:
    """メインエントリーポイント."""
    args = parse_args()

    if args.command == "train":
        config = ConfigLoader.load(args.config)
        train(config, args.config)
    elif args.command == "infer":
        config = ConfigLoader.load(args.config)
        infer(config, args.image, args.threshold, args.model_dir)
    else:
        # コマンド未指定の場合はヘルプを表示
        parse_args()


if __name__ == "__main__":
    main()
