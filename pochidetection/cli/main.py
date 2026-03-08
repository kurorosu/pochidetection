"""物体検出 CLI.

RT-DETR / SSDLite モデルの学習・推論・ONNXエクスポートを行うコマンドラインインターフェース.

使用方法:
    uv run pochi train
    uv run pochi train -c configs/rtdetr_coco.py
    uv run pochi train -c configs/ssdlite_coco.py
    uv run pochi infer -d images/
    uv run pochi infer -d images/ -m work_dirs/20260124_001/best
    uv run pochi export -m work_dirs/20260124_001/best
"""

import argparse
import sys
from collections.abc import Callable
from typing import Any

from pochidetection.logging import LoggerManager, LogLevel
from pochidetection.utils import ConfigLoader
from pochidetection.utils.config_resolver import resolve_config_path

DEFAULT_CONFIG = "configs/rtdetr_coco.py"


def _create_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサーを構築する.

    Returns:
        構築した ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="物体検出モデルの学習・推論・エクスポート CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  学習 (RT-DETR):
    uv run pochi train
    uv run pochi train -c configs/rtdetr_coco.py

  学習 (SSDLite):
    uv run pochi train -c configs/ssdlite_coco.py

  推論 (RT-DETR):
    uv run pochi infer -d images/
    uv run pochi infer -d images/ -m work_dirs/20260124_001/best
    uv run pochi infer -d images/ -m model.engine

  推論 (SSDLite):
    uv run pochi infer -d images/ -c configs/ssdlite_coco.py
    uv run pochi infer -d images/ -m work_dirs/20260124_001/best -c configs/ssdlite_coco.py

  ONNXエクスポート (RT-DETR のみ):
    uv run pochi export -m work_dirs/20260124_001/best
    uv run pochi export -m work_dirs/20260124_001/best -o model.onnx
    uv run pochi export -m work_dirs/20260124_001/best --input-size 640 640

  TensorRTエクスポート (RT-DETR のみ, FP32):
    uv run pochi export-trt -i model.onnx
    uv run pochi export-trt -i model.onnx --max-batch 8

  TensorRTエクスポート (RT-DETR のみ, FP16):
    uv run pochi export-trt -i model.onnx --fp16
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
        default=None,
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
        default=None,
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
        help="出力エンジンパス (default: model_fp32.engine / model_fp16.engine)",
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
        "--fp16",
        action="store_true",
        help="FP16 精度でエンジンをビルド (非対応GPUではFP32にフォールバック)",
    )
    export_trt_parser.add_argument(
        "--build-memory",
        type=int,
        default=4 * 1024 * 1024 * 1024,
        help="TensorRT ビルド時の最大メモリ (bytes, default: 4GB)",
    )
    export_trt_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help=f"設定ファイルのパス (default: {DEFAULT_CONFIG})",
    )

    return parser


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース.

    Returns:
        パースされた引数.
    """
    return _create_parser().parse_args()


def setup_logging(debug: bool = False) -> None:
    """ログ設定の初期化.

    Args:
        debug: DEBUGモードを有効にするか.
    """
    logger_manager = LoggerManager()
    level = LogLevel.DEBUG if debug else LogLevel.INFO
    logger_manager.set_default_level(level)


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


def _resolve_infer(
    config: dict[str, Any],
) -> Callable[[dict[str, Any], str, str | None], None]:
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


def main() -> None:
    """メインエントリーポイント."""
    args = parse_args()
    setup_logging(debug=args.debug)

    if args.command == "train":
        config = ConfigLoader.load(args.config)
        train_fn = _resolve_train(config)
        train_fn(config, args.config)
    elif args.command == "infer":
        config_path = resolve_config_path(args.config, args.model_dir, DEFAULT_CONFIG)
        config = ConfigLoader.load(config_path)
        infer_fn = _resolve_infer(config)
        infer_fn(config, args.dir, args.model_dir)
    elif args.command == "export":
        config_path = resolve_config_path(args.config, args.model_dir, DEFAULT_CONFIG)
        config = ConfigLoader.load(config_path)
        if config.get("architecture") == "SSDLite":
            print(
                "Error: export コマンドは SSDLite に対応していません. "
                "RT-DETR の設定ファイルを指定してください.",
                file=sys.stderr,
            )
            sys.exit(1)

        from pochidetection.scripts.rtdetr.export_onnx import export_onnx

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
        config_path = resolve_config_path(args.config, args.onnx_path, DEFAULT_CONFIG)
        config = ConfigLoader.load(config_path)
        if config.get("architecture") == "SSDLite":
            print(
                "Error: export-trt コマンドは SSDLite に対応していません. "
                "RT-DETR の設定ファイルを指定してください.",
                file=sys.stderr,
            )
            sys.exit(1)

        from pochidetection.scripts.rtdetr.export_trt import export_trt

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
            args.fp16,
            args.build_memory,
        )
    else:
        _create_parser().print_help()


if __name__ == "__main__":
    main()
