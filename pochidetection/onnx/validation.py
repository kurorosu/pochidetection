"""ONNX エクスポート検証ユーティリティ."""

import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from pochidetection.logging import LoggerManager

logger: logging.Logger = LoggerManager().get_logger(__name__)


def verify_onnx_outputs(
    onnx_path: Path,
    pytorch_outputs: list[np.ndarray],
    dummy_input: np.ndarray,
    output_names: list[str],
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> bool:
    """ONNX モデルの構造検証と PyTorch 出力との比較を行う.

    Args:
        onnx_path: ONNX モデルのパス.
        pytorch_outputs: 比較対象の PyTorch 出力 (numpy 配列のリスト).
        dummy_input: ONNX Runtime に入力する numpy 配列.
        output_names: ログ表示用の出力名リスト.
        rtol: 相対許容誤差.
        atol: 絶対許容誤差.

    Returns:
        全出力が許容誤差内で一致する場合 True.
    """
    logger.debug("ONNXモデルの構造を検証中...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.debug("構造検証: OK")

    logger.debug("PyTorchとONNXの出力を比較中...")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_results = session.run(None, {"pixel_values": dummy_input})

    is_close = True
    diffs: list[str] = []
    for pt_out, onnx_out, name in zip(pytorch_outputs, onnx_results, output_names):
        onnx_out_f32 = onnx_out.astype(np.float32)
        close: bool = bool(np.allclose(pt_out, onnx_out_f32, rtol=rtol, atol=atol))
        if not close:
            is_close = False
        max_diff = float(np.max(np.abs(pt_out - onnx_out_f32)))
        diffs.append(f"{name}: {max_diff:.2e}")

    diff_str = ", ".join(diffs)
    if is_close:
        logger.info("出力比較: OK")
        logger.debug(f"最大差分 - {diff_str}")
    else:
        logger.warning("出力比較: NG")
        logger.warning(f"最大差分 - {diff_str}")

    return is_close
