"""TensorRT エクスポートスクリプト (アーキテクチャ共通)."""

import logging
import sys
from pathlib import Path

from pochidetection.logging import LoggerManager
from pochidetection.tensorrt import DEFAULT_BUILD_MEMORY, TensorRTExporter

logger: logging.Logger = LoggerManager().get_logger(__name__)


def export_trt(
    onnx_path_str: str,
    output_path_str: str | None,
    input_size: tuple[int, int],
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 4,
    use_fp16: bool = False,
    use_int8: bool = False,
    int8_calibrator: object | None = None,
    build_memory: int = DEFAULT_BUILD_MEMORY,
) -> None:
    """Execute TensorRT export from an ONNX model.

    Args:
        onnx_path_str: 入力ONNXモデルのパス文字列.
        output_path_str: 出力エンジンファイル (.engine) のパス文字列.
            None の場合は ONNX ファイルと同じディレクトリに配置される.
        input_size: 入力サイズ (height, width).
        min_batch: 最小バッチサイズ.
        opt_batch: 最適バッチサイズ.
        max_batch: 最大バッチサイズ.
        use_fp16: FP16 精度でビルドするかどうか.
        use_int8: INT8 精度でビルドするかどうか (PTQ).
        int8_calibrator: INT8 キャリブレータ. use_int8=True の場合に必要.
        build_memory: TensorRT ビルド時のメモリプール制限 (bytes).
    """
    logger.info("TensorRTエクスポートを開始します")

    onnx_path = Path(onnx_path_str)

    if output_path_str is None:
        if use_int8:
            engine_name = "model_int8.engine"
        elif use_fp16:
            engine_name = "model_fp16.engine"
        else:
            engine_name = "model_fp32.engine"
        output_path = onnx_path.parent / engine_name
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
            use_fp16=use_fp16,
            use_int8=use_int8,
            int8_calibrator=int8_calibrator,
            build_memory=build_memory,
        )
    except (OSError, RuntimeError) as e:
        logger.error(f"TensorRTエクスポートに失敗しました: {e}")
        sys.exit(1)

    logger.info("TensorRTエクスポート処理が完了しました")
