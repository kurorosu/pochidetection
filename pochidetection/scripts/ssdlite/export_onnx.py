"""SSDLite モデルの ONNX エクスポートスクリプト."""

import logging
import sys
from pathlib import Path

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.logging import LoggerManager
from pochidetection.onnx import SSDLiteOnnxExporter

logger: logging.Logger = LoggerManager().get_logger(__name__)


def export_onnx(
    config: DetectionConfigDict,
    model_dir: str,
    output: str | None,
    opset_version: int,
    input_size: tuple[int, int] | None,
    skip_verify: bool,
    fp16: bool,
) -> None:
    """ONNX エクスポートを実行 (SSDLite).

    Args:
        config: 設定辞書.
        model_dir: モデルディレクトリのパス.
        output: 出力ファイルパス. None の場合は model_dir 内に model.onnx を出力.
        opset_version: ONNX オペセットバージョン.
        input_size: 入力サイズ (height, width). None の場合は config から取得.
        skip_verify: エクスポート後の検証をスキップするか.
        fp16: FP16 でエクスポートするか.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.error(f"モデルディレクトリが見つかりません: {model_path}")
        sys.exit(1)

    if output is not None:
        output_path = Path(output)
    else:
        suffix = "_fp16" if fp16 else "_fp32"
        output_path = model_path / f"model{suffix}.onnx"

    if input_size is not None:
        height, width = input_size
    else:
        image_size = config.get("image_size", {})
        height = image_size.get("height", 320)
        width = image_size.get("width", 320)

    logger.debug(f"モデルディレクトリ: {model_path}")
    logger.debug(f"入力サイズ: {height}x{width}")
    logger.debug(f"出力先: {output_path}")
    logger.debug(f"opset_version: {opset_version}")
    logger.debug(f"FP16: {fp16}")

    num_classes = config["num_classes"]
    nms_iou_threshold = config.get("nms_iou_threshold", 0.5)

    exporter = SSDLiteOnnxExporter()

    try:
        exporter.load_model(
            model_path,
            num_classes=num_classes,
            nms_iou_threshold=nms_iou_threshold,
        )
    except Exception as e:
        logger.error(f"モデルの読み込みに失敗: {e}")
        sys.exit(1)

    try:
        logger.info("ONNX変換を実行中...")
        exporter.export(
            output_path=output_path,
            input_size=(height, width),
            opset_version=opset_version,
            fp16=fp16,
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
                fp16=fp16,
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
