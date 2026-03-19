"""推論バックエンドの同期ユーティリティのテスト."""

import torch
import torch.nn as nn

from pochidetection.inference.sync import synchronize_cuda


class TestSynchronizeCuda:
    """synchronize_cuda のテスト."""

    def test_cpu_model_does_not_raise(self) -> None:
        """CPU モデルに対して例外が発生しない."""
        model = nn.Linear(4, 2)
        synchronize_cuda(model)

    def test_cpu_model_device_is_cpu(self) -> None:
        """CPU モデルのデバイスが cpu であることを確認."""
        model = nn.Linear(4, 2)
        device = next(model.parameters()).device
        assert device.type == "cpu"
        # CPU の場合は synchronize_cuda が何もしないことを確認
        synchronize_cuda(model)

    def test_function_is_idempotent(self) -> None:
        """複数回呼び出しても例外が発生しない."""
        model = nn.Linear(4, 2)
        synchronize_cuda(model)
        synchronize_cuda(model)
        synchronize_cuda(model)
