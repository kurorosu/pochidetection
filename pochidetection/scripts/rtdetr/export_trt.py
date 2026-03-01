"""TensorRTエクスポートスクリプト."""

import logging
import sys
from pathlib import Path

from pochidetection.logging import LoggerManager
from pochidetection.tensorrt import TensorRTExporter

logger: logging.Logger = LoggerManager().get_logger(__name__)


def export_trt(
    onnx_path_str: str,
    output_path_str: str | None,
    input_size: tuple[int, int],
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 4,
) -> None:
    """TensorRTエクスポートを実行.

    Args:
        onnx_path_str: 入力ONNXモデルのパス文字列.
        output_path_str: 出力エンジンファイル (.engine) のパス文字列.
            None の場合は ONNX ファイルと同じディレクトリに配置される.
        input_size: 入力サイズ (height, width).
        min_batch: 最小バッチサイズ.
        opt_batch: 最適バッチサイズ.
        max_batch: 最大バッチサイズ.
    """
    logger.info("TensorRTエクスポートを開始します")

    onnx_path = Path(onnx_path_str)

    if output_path_str is None:
        output_path = onnx_path.with_suffix(".engine")
    else:
        output_path = Path(output_path_str)

    try:
        exporter = TensorRTExporter()
        exporter.export(
            onnx_path=onnx_path,
            output_path=output_path,
            input_size=input_size,
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
        )
    except Exception as e:
        logger.error(f"TensorRTエクスポートに失敗しました: {e}")
        sys.exit(1)

    logger.info("TensorRTエクスポート処理が完了しました")
