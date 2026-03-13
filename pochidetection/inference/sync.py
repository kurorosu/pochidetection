"""推論バックエンドの同期ユーティリティ."""

import torch
import torch.nn as nn


def synchronize_cuda(model: nn.Module) -> None:
    """モデルが CUDA 上にある場合に同期する.

    Args:
        model: パラメータからデバイスを判定する PyTorch モデル.
    """
    if torch.cuda.is_available():
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
