"""TensorRT メモリ管理ユーティリティのテスト."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("tensorrt")

import tensorrt as trt

from pochidetection.tensorrt.rtdetr.memory import (
    TRT_TO_NUMPY_DTYPE,
    TRT_TO_TORCH_DTYPE,
    TensorBinding,
    allocate_bindings,
)


class TestTensorBinding:
    """TensorBinding データクラスのテスト."""

    def test_fields(self) -> None:
        """フィールドが正しく保持されることを確認."""
        tensor = torch.empty(1, 3, 64, 64, device="cuda")
        binding = TensorBinding(
            name="pixel_values",
            shape=(1, 3, 64, 64),
            numpy_dtype=np.dtype(np.float32),
            torch_dtype=torch.float32,
            is_input=True,
            device_tensor=tensor,
        )

        assert binding.name == "pixel_values"
        assert binding.shape == (1, 3, 64, 64)
        assert binding.numpy_dtype == np.dtype(np.float32)
        assert binding.torch_dtype == torch.float32
        assert binding.is_input is True
        assert binding.device_tensor is tensor


class TestDtypeMappings:
    """データ型マッピングのテスト."""

    def test_numpy_dtype_mapping_contains_float32(self) -> None:
        """float32 の NumPy マッピングが存在することを確認."""
        assert trt.float32 in TRT_TO_NUMPY_DTYPE
        assert TRT_TO_NUMPY_DTYPE[trt.float32] == np.dtype(np.float32)

    def test_torch_dtype_mapping_contains_float32(self) -> None:
        """float32 の PyTorch マッピングが存在することを確認."""
        assert trt.float32 in TRT_TO_TORCH_DTYPE
        assert TRT_TO_TORCH_DTYPE[trt.float32] == torch.float32

    def test_numpy_dtype_mapping_contains_float16(self) -> None:
        """float16 の NumPy マッピングが存在することを確認."""
        assert trt.float16 in TRT_TO_NUMPY_DTYPE
        assert TRT_TO_NUMPY_DTYPE[trt.float16] == np.dtype(np.float16)

    def test_torch_dtype_mapping_contains_float16(self) -> None:
        """float16 の PyTorch マッピングが存在することを確認."""
        assert trt.float16 in TRT_TO_TORCH_DTYPE
        assert TRT_TO_TORCH_DTYPE[trt.float16] == torch.float16


class TestAllocateBindings:
    """allocate_bindings 関数のテスト."""

    def test_returns_bindings_list(self, engine_path: Path) -> None:
        """バインディングリストが返されることを確認."""
        trt_logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        # 動的バッチの shape を確定させる.
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                shape = list(engine.get_tensor_shape(name))
                if shape[0] == -1:
                    shape[0] = 1
                    context.set_input_shape(name, shape)

        bindings = allocate_bindings(engine, context)

        assert isinstance(bindings, list)
        assert len(bindings) > 0

    def test_bindings_have_correct_attributes(self, engine_path: Path) -> None:
        """各バインディングが正しい属性を持つことを確認."""
        trt_logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                shape = list(engine.get_tensor_shape(name))
                if shape[0] == -1:
                    shape[0] = 1
                    context.set_input_shape(name, shape)

        bindings = allocate_bindings(engine, context)

        for binding in bindings:
            assert isinstance(binding, TensorBinding)
            assert isinstance(binding.name, str)
            assert isinstance(binding.shape, tuple)
            assert isinstance(binding.is_input, bool)
            assert binding.device_tensor.is_cuda

    def test_bindings_contain_input_and_output(self, engine_path: Path) -> None:
        """入力と出力の両方のバインディングが含まれることを確認."""
        trt_logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                shape = list(engine.get_tensor_shape(name))
                if shape[0] == -1:
                    shape[0] = 1
                    context.set_input_shape(name, shape)

        bindings = allocate_bindings(engine, context)

        has_input = any(b.is_input for b in bindings)
        has_output = any(not b.is_input for b in bindings)
        assert has_input
        assert has_output
