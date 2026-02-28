"""RT-DETR CLI.

RT-DETRモデルの学習・推論・ONNXエクスポートを行うコマンドラインインターフェース.

使用方法:
    uv run pochidet-rtdetr train
    uv run pochidet-rtdetr train -c configs/rtdetr_coco.py
    uv run pochidet-rtdetr infer -d images/
    uv run pochidet-rtdetr infer -d images/ -t 0.3 -m work_dirs/20260124_001/best
    uv run pochidet-rtdetr export -m work_dirs/20260124_001/best
"""

import argparse

from pochidetection.scripts.rtdetr.export_onnx import export_onnx
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
    uv run pochidet-rtdetr infer -d images/
    uv run pochidet-rtdetr infer -d images/ -t 0.3
    uv run pochidet-rtdetr infer -d images/ -m work_dirs/20260124_001/best

  ONNXエクスポート:
    uv run pochidet-rtdetr export -m work_dirs/20260124_001/best
    uv run pochidet-rtdetr export -m work_dirs/20260124_001/best -o model.onnx
    uv run pochidet-rtdetr export -m work_dirs/20260124_001/best --input-size 640 640
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
    infer_parser = subparsers.add_parser("infer", help="フォルダ内画像の一括推論")
    infer_parser.add_argument(
        "-d",
        "--dir",
        type=str,
        required=True,
        help="推論対象の画像フォルダパス",
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

    # ONNXエクスポートコマンド
    export_parser = subparsers.add_parser(
        "export", help="学習済みモデルのONNXエクスポート"
    )
    export_parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        required=True,
        help="モデルディレクトリのパス",
    )
    export_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="出力ファイルパス (default: <model_dir>/model.onnx)",
    )
    export_parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="入力画像サイズ (default: configのimage_sizeを使用)",
    )
    export_parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNXオペセットバージョン (default: 17)",
    )
    export_parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="エクスポート後の検証をスキップ",
    )
    export_parser.add_argument(
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
        infer(config, args.dir, args.threshold, args.model_dir)
    elif args.command == "export":
        config = ConfigLoader.load(args.config)
        input_size = tuple(args.input_size) if args.input_size else None
        export_onnx(
            config,
            args.model_dir,
            args.output,
            args.opset_version,
            input_size,
            args.skip_verify,
        )
    else:
        # コマンド未指定の場合はヘルプを表示
        parse_args()


if __name__ == "__main__":
    main()
