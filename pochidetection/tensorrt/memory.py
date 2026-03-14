"""TensorRT 推論用 CUDA メモリ管理ユーティリティ.

TensorRT エンジンの入出力テンソルに対して,
GPU メモリの確保・アドレス設定・ホスト転送を行う.
メモリ管理には PyTorch の CUDA テンソルを利用し,
cuda-python (cudart) への依存を回避する.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

try:
    import tensorrt as trt

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False

if TYPE_CHECKING:
    import tensorrt as trt

from pochidetection.logging import LoggerManager

logger: logging.Logger = LoggerManager().get_logger(__name__)

TRT_TO_NUMPY_DTYPE: dict[object, np.dtype[np.generic]] = {}
TRT_TO_TORCH_DTYPE: dict[object, torch.dtype] = {}

if _TRT_AVAILABLE:
    TRT_TO_NUMPY_DTYPE = {
        trt.float32: np.dtype(np.float32),
        trt.float16: np.dtype(np.float16),
        trt.int32: np.dtype(np.int32),
        trt.int8: np.dtype(np.int8),
    }
    TRT_TO_TORCH_DTYPE = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
    }


@dataclass
class TensorBinding:
    """TensorRT I/O テンソルのバインディング情報.

    Attributes:
        name: テンソル名.
        shape: テンソルの形状.
        numpy_dtype: NumPy データ型.
        torch_dtype: PyTorch データ型.
        is_input: 入力テンソルかどうか.
        device_tensor: GPU 上の PyTorch テンソル.
    """

    name: str
    shape: tuple[int, ...]
    numpy_dtype: np.dtype[np.generic]
    torch_dtype: torch.dtype
    is_input: bool
    device_tensor: torch.Tensor = field(repr=False)


def allocate_bindings(
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
) -> list[TensorBinding]:
    """Tensorrt エンジンの I/O テンソルに対して GPU メモリを確保する.

    PyTorch の CUDA テンソルを利用して GPU メモリを確保し,
    そのアドレスを TensorRT の ExecutionContext に設定する.

    Args:
        engine: TensorRT の ICudaEngine インスタンス.
        context: TensorRT の IExecutionContext インスタンス.

    Returns:
        確保した TensorBinding のリスト.

    Raises:
        ImportError: tensorrt がインストールされていない場合.
        ValueError: 未対応の TensorRT データ型が検出された場合.
    """
    if not _TRT_AVAILABLE:
        raise ImportError(
            "tensorrt パッケージがインストールされていません. "
            "GPU環境構築手順に従って TensorRT をインストールしてください."
        )

    bindings: list[TensorBinding] = []

    for i in range(engine.num_io_tensors):
        name: str = engine.get_tensor_name(i)
        trt_dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        is_input = mode == trt.TensorIOMode.INPUT

        shape = tuple(context.get_tensor_shape(name))
        if trt_dtype not in TRT_TO_NUMPY_DTYPE:
            raise ValueError(
                f"未対応の TensorRT データ型です: {trt_dtype} (テンソル: {name})"
            )

        numpy_dtype = TRT_TO_NUMPY_DTYPE[trt_dtype]
        torch_dtype = TRT_TO_TORCH_DTYPE[trt_dtype]

        device_tensor = torch.empty(shape, dtype=torch_dtype, device="cuda")
        context.set_tensor_address(name, device_tensor.data_ptr())

        kind = "input" if is_input else "output"
        logger.debug(
            f"Allocated {kind} tensor '{name}': " f"shape={shape}, dtype={numpy_dtype}"
        )

        bindings.append(
            TensorBinding(
                name=name,
                shape=shape,
                numpy_dtype=numpy_dtype,
                torch_dtype=torch_dtype,
                is_input=is_input,
                device_tensor=device_tensor,
            )
        )

    return bindings
