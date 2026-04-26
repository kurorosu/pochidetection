"""推論バックエンド構築ヘルパー.

PyTorch 推論バックエンドの共通構築後処理 (device 配置, eval, FP16) と
backend ラップを集約し, アーキテクチャ別 ``_create_pytorch_backend`` の
boilerplate を削減する.
"""

from collections.abc import Callable
from typing import TypeVar

from pochidetection.interfaces.model import IDetectionModel

__all__ = ["build_pytorch_backend"]


T = TypeVar("T")


def build_pytorch_backend(
    model: IDetectionModel,
    backend_cls: Callable[..., T],
    device: str,
    use_fp16: bool,
) -> T:
    """構築済みモデルから PyTorch 推論バックエンドを組み立てる.

    Args:
        model: 構築済み (重みロード済み) の検出モデル.
        backend_cls: バックエンドのクラス (model のみを受け取る __init__ を想定).
        device: 推論デバイス.
        use_fp16: True で FP16 化する.

    Returns:
        backend_cls(model) のインスタンス.
    """
    model.to(device)
    model.eval()
    if use_fp16:
        model.half()
    return backend_cls(model)
