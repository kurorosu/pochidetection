"""CLI 引数パーサー定義.

_create_parser() でサブコマンドを構築し, parse_args() で引数をパースする.
"""

import argparse

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
使用例 (-c で config を切り替えると RT-DETR / SSDLite が自動選択されます):
  学習:
    uv run pochi train                                        # RT-DETR (default)
    uv run pochi train -c configs/ssdlite_coco.py             # SSDLite

  推論 (-m のモデルフォルダ内 config で自動判定):
    uv run pochi infer
    uv run pochi infer -m work_dirs/20260124_001/best
    uv run pochi infer -m work_dirs/20260310_002/best/model_fp32.onnx
    uv run pochi infer -m work_dirs/20260310_002/best/model_fp32.engine

  動画推論:
    uv run pochi infer -d video.mp4 -m work_dirs/20260124_001/best
    uv run pochi infer -d video.mp4 --interval 3                     # 3フレーム間隔
    uv run pochi infer -d video.mp4                                  # COCO プリトレイン

  リアルタイム推論 (Webcam / RTSP):
    uv run pochi infer -d 0                                          # Webcam
    uv run pochi infer -d 0 --record output.mp4                      # Webcam + 録画
    uv run pochi infer -d rtsp://192.168.1.10/stream                 # RTSP
    uv run pochi infer -d rtsp://... --record out.mp4                # RTSP + 録画

  エクスポート (入力パスで ONNX / TensorRT を自動判定):
    uv run pochi export -m work_dirs/20260124_001/best                         # フォルダ → ONNX
    uv run pochi export -m work_dirs/20260124_001/best --fp16                  # SSDLite FP16 ONNX
    uv run pochi export -m work_dirs/20260124_001/best/model_fp32.onnx         # .onnx → TensorRT
    uv run pochi export -m work_dirs/20260124_001/best/model_fp32.onnx --fp16  # TensorRT FP16
        """,
    )

    # グローバルオプション
    parser.add_argument("--debug", action="store_true", help="DEBUGログを有効化")

    # サブコマンド
    subparsers = parser.add_subparsers(
        dest="command",
        help="実行するコマンド",
        required=True,
    )

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
    infer_parser = subparsers.add_parser(
        "infer", help="画像フォルダ, 動画ファイル, Webcam, RTSP の推論"
    )
    infer_parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default=None,
        help="推論対象 (画像フォルダ / 動画ファイル / Webcam デバイスID / RTSP URL). "
        "未指定時は config の infer_image_dir を使用",
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
    infer_parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="N フレーム間隔で推論 (動画/ストリームのみ, default: 1 = 全フレーム)",
    )
    infer_parser.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="OUTPUT.mp4",
        help="ストリーム推論時に表示と同時に録画する出力ファイルパス",
    )

    # エクスポートコマンド (フォルダ → ONNX, .onnx → TensorRT を自動判定)
    export_parser = subparsers.add_parser(
        "export",
        help="モデルのエクスポート (フォルダ → ONNX, .onnx → TensorRT)",
    )
    export_parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        required=True,
        help="モデルディレクトリ (→ ONNX) または ONNXファイル (→ TensorRT)",
    )
    export_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="出力ファイルパス (default: 自動決定)",
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
        help="ONNXオペセットバージョン (default: 17, ONNX時のみ)",
    )
    export_parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="エクスポート後の検証をスキップ (ONNX時のみ)",
    )
    export_parser.add_argument(
        "--fp16",
        action="store_true",
        help="FP16 精度でエクスポート (ONNX: SSDLiteのみ, TRT: 全アーキテクチャ)",
    )
    export_parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="最小バッチサイズ (TRT時のみ, default: 1)",
    )
    export_parser.add_argument(
        "--opt-batch",
        type=int,
        default=1,
        help="最適バッチサイズ (TRT時のみ, default: 1)",
    )
    export_parser.add_argument(
        "--max-batch",
        type=int,
        default=4,
        help="最大バッチサイズ (TRT時のみ, default: 4)",
    )
    export_parser.add_argument(
        "--build-memory",
        type=int,
        default=4 * 1024 * 1024 * 1024,
        help="TensorRT ビルド時の最大メモリ (bytes, default: 4GB, TRT時のみ)",
    )
    export_parser.add_argument(
        "--int8",
        action="store_true",
        help="INT8 精度でエクスポート (PTQ, TRT時のみ, キャリブレーションデータは config から取得)",
    )
    export_parser.add_argument(
        "--calib-max-images",
        type=int,
        default=None,
        help="キャリブレーションに使用する最大画像数 (default: 全件, INT8時のみ)",
    )
    export_parser.add_argument(
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
