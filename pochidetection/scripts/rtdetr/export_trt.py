"""TensorRTエクスポートスクリプト."""

import logging
import sys
from pathlib import Path

from pochidetection.logging import LoggerManager
from pochidetection.tensorrt import DEFAULT_BUILD_MEMORY, RTDetrTensorRTExporter

logger: logging.Logger = LoggerManager().get_logger(__name__)


def export_trt(
    onnx_path_str: str,
    output_path_str: str | None,
    input_size: tuple[int, int],
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 4,
    use_fp16: bool = False,
    build_memory: int = DEFAULT_BUILD_MEMORY,
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
        use_fp16: FP16 精度でビルドするかどうか.
        build_memory: TensorRT ビルド時のメモリプール制限 (bytes).
    """
    logger.info("TensorRTエクスポートを開始します")

    onnx_path = Path(onnx_path_str)

    if output_path_str is None:
        engine_name = "model_fp16.engine" if use_fp16 else "model_fp32.engine"
        output_path = onnx_path.parent / engine_name
    else:
        output_path = Path(output_path_str)

    try:
        exporter = RTDetrTensorRTExporter()
        exporter.export(
            onnx_path=onnx_path,
            output_path=output_path,
            input_size=input_size,
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
            use_fp16=use_fp16,
            build_memory=build_memory,
        )
    except Exception as e:
        logger.error(f"TensorRTエクスポートに失敗しました: {e}")
        sys.exit(1)

    logger.info("TensorRTエクスポート処理が完了しました")
