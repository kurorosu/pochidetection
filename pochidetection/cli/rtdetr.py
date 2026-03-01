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

from pochidetection.logging import LoggerManager, LogLevel
from pochidetection.scripts.rtdetr.export_onnx import export_onnx
from pochidetection.scripts.rtdetr.export_trt import export_trt
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
    uv run pochidet-rtdetr infer -d images/ -m model.engine

  ONNXエクスポート:
    uv run pochidet-rtdetr export -m work_dirs/20260124_001/best
    uv run pochidet-rtdetr export -m work_dirs/20260124_001/best -o model.onnx
    uv run pochidet-rtdetr export -m work_dirs/20260124_001/best --input-size 640 640

  TensorRTエクスポート (FP32):
    uv run pochidet-rtdetr export-trt -i model.onnx
    uv run pochidet-rtdetr export-trt -i model.onnx --max-batch 8
        """,
    )

    # グローバルオプション
    parser.add_argument("--debug", action="store_true", help="DEBUGログを有効化")

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

    # TensorRTエクスポートコマンド
    export_trt_parser = subparsers.add_parser(
        "export-trt", help="ONNXモデルからTensorRTエンジンへのエクスポート"
    )
    export_trt_parser.add_argument(
        "-i", "--onnx-path", type=str, required=True, help="入力ONNXモデルのパス"
    )
    export_trt_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="出力エンジンパス (default: <onnx_path>.engine)",
    )
    export_trt_parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="入力画像サイズ (default: configのimage_sizeを使用)",
    )
    export_trt_parser.add_argument(
        "--min-batch", type=int, default=1, help="最小バッチサイズ (default: 1)"
    )
    export_trt_parser.add_argument(
        "--opt-batch", type=int, default=1, help="最適バッチサイズ (default: 1)"
    )
    export_trt_parser.add_argument(
        "--max-batch", type=int, default=4, help="最大バッチサイズ (default: 4)"
    )
    export_trt_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"設定ファイルのパス (default: {DEFAULT_CONFIG})",
    )

    return parser.parse_args()


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
    elif args.command == "export-trt":
        config = ConfigLoader.load(args.config)
        input_size_tgt: tuple[int, int] = (
            (args.input_size[0], args.input_size[1])
            if args.input_size
            else (
                int(config["image_size"]["height"]),
                int(config["image_size"]["width"]),
            )
        )
        export_trt(
            args.onnx_path,
            args.output,
            input_size_tgt,
            args.min_batch,
            args.opt_batch,
            args.max_batch,
        )
    else:
        # コマンド未指定の場合はヘルプを表示
        parse_args()


if __name__ == "__main__":
    main()
