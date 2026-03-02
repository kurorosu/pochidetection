"""RT-DETRモデルのONNXエクスポートスクリプト."""

import logging
import sys
from pathlib import Path
from typing import Any

from pochidetection.logging import LoggerManager
from pochidetection.onnx import OnnxExporter

logger: logging.Logger = LoggerManager().get_logger(__name__)


def export_onnx(
    config: dict[str, Any],
    model_dir: str,
    output: str | None,
    opset_version: int,
    input_size: tuple[int, int] | None,
    skip_verify: bool,
) -> None:
    """ONNXエクスポートを実行.

    Args:
        config: 設定辞書.
        model_dir: モデルディレクトリのパス.
        output: 出力ファイルパス. Noneの場合はmodel_dir内にmodel.onnxを出力.
        opset_version: ONNXオペセットバージョン.
        input_size: 入力サイズ (height, width). Noneの場合はconfigから取得.
        skip_verify: エクスポート後の検証をスキップするか.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.error(f"モデルディレクトリが見つかりません: {model_path}")
        sys.exit(1)

    if output is not None:
        output_path = Path(output)
    else:
        output_path = model_path / "model_fp32.onnx"

    if input_size is not None:
        height, width = input_size
    else:
        image_size = config.get("image_size", {})
        height = image_size.get("height", 640)
        width = image_size.get("width", 640)

    logger.debug(f"モデルディレクトリ: {model_path}")
    logger.debug(f"入力サイズ: {height}x{width}")
    logger.debug(f"出力先: {output_path}")
    logger.debug(f"opset_version: {opset_version}")

    exporter = OnnxExporter()

    try:
        exporter.load_model(model_path)
    except Exception as e:
        logger.error(f"モデルの読み込みに失敗: {e}")
        sys.exit(1)

    try:
        logger.info("ONNX変換を実行中...")
        exporter.export(
            output_path=output_path,
            input_size=(height, width),
            opset_version=opset_version,
        )
    except Exception as e:
        logger.error(f"ONNX変換に失敗: {e}")
        sys.exit(1)

    if not skip_verify:
        logger.info("--- ONNX検証 ---")
        try:
            is_valid = exporter.verify(
                onnx_path=output_path,
                input_size=(height, width),
            )
            if is_valid:
                logger.info("検証完了: ONNXモデルは正常です")
            else:
                logger.warning("PyTorchとONNXの出力に差異があります")
                sys.exit(1)
        except Exception as e:
            logger.error(f"ONNX検証に失敗: {e}")
            sys.exit(1)
    else:
        logger.info("検証をスキップしました")
