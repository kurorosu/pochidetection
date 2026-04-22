"""推論 / 学習の実行環境 (device / mode / cudnn) 解決ユーティリティ.

backend 種別 (PyTorch / ONNX / TensorRT) と config から, 推論 pipeline が使う
実効デバイスや preprocess の経路 (cpu/gpu), cudnn.benchmark の有効化などを解決する.

公開関数:

- ``resolve_device``: モデル形式から ``(actual_device, runtime_device)`` を解決.
- ``resolve_pipeline_mode``: preprocess 経路 ('cpu' / 'gpu') を解決.
- ``setup_cudnn_benchmark``: config に従い ``torch.backends.cudnn.benchmark`` を有効化.
"""

from pathlib import Path
from typing import Any, Literal

import torch

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.logging import LoggerManager
from pochidetection.pipelines.model_path import is_onnx_model, is_tensorrt_model

__all__ = [
    "resolve_device",
    "resolve_pipeline_mode",
    "setup_cudnn_benchmark",
]

logger = LoggerManager().get_logger(__name__)


def resolve_pipeline_mode(
    requested: Literal["cpu", "gpu"] | None,
    model_path: Path,
) -> Literal["cpu", "gpu"]:
    """Preprocess の経路を backend 種別から解決する.

    PyTorch / TensorRT は default 'gpu', ONNX は default 'cpu'.
    ONNX で 'gpu' を明示指定された場合は ValueError で起動を拒否する
    (ONNX Runtime は CPU numpy を要求するため GPU preprocess の効果がないため).

    Args:
        requested: CLI / config で指定された値. None の場合は backend 種別から決定.
        model_path: モデルのパス (backend 種別判定用).

    Returns:
        解決後の経路名 ('cpu' or 'gpu').

    Raises:
        ValueError: ONNX backend で 'gpu' を明示指定された場合.
    """
    if is_onnx_model(model_path):
        if requested == "gpu":
            raise ValueError(
                "ONNX backend は --pipeline cpu のみ対応. "
                "フォールバック手順: "
                "(1) `--pipeline cpu` を指定するか未指定にする, "
                "または (2) GPU preprocess を使う場合は PyTorch (.pth) / "
                "TensorRT (.engine) バックエンドのモデルに切り替える."
            )
        return "cpu"
    return requested if requested is not None else "gpu"


def setup_cudnn_benchmark(config: DetectionConfigDict) -> None:
    """cudnn.benchmark を設定する.

    Args:
        config: 設定辞書.
    """
    device = config["device"]
    if config.get("cudnn_benchmark", False) and device == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cudnn.benchmark enabled")


def resolve_device(
    model_path: Path,
    config: DetectionConfigDict,
    backend: IInferenceBackend[Any],
) -> tuple[str, str]:
    """モデル形式に応じたデバイスを解決する.

    backend 種別ごとに以下のように決定する:

    - TensorRT (.engine): ``("cuda", "cuda")`` 固定. 推論も preprocess も GPU.
    - ONNX (.onnx): ``(actual_device, "cpu")``.
      ``actual_device`` は ``backend.active_providers`` に
      ``CUDAExecutionProvider`` が含まれるかで ``"cuda"`` / ``"cpu"`` を判定する
      (ログ表示・メトリクス用). 一方 ``runtime_device`` は常に ``"cpu"``
      とする. ONNX Runtime の ``session.run()`` は入力を CPU 上の numpy
      配列で受け取るため, preprocess 結果を GPU テンソルで渡しても
      Runtime 側で CPU へコピーし直され無駄が生じる. このため
      preprocess の配置先 (=``runtime_device``) を CPU に固定し,
      GPU preprocess 経路を選ばせないようにしている.
    - PyTorch (.pth): ``(device, device)``. config の ``device`` をそのまま使う.

    Args:
        model_path: モデルのパス.
        config: 設定辞書.
        backend: 生成済みのバックエンド. ONNX の場合のみ
            ``active_providers`` 属性を参照する.

    Returns:
        ``(actual_device, runtime_device)`` のタプル.

        - ``actual_device``: 推論が実際に走るデバイス (ログ / メトリクス用).
        - ``runtime_device``: preprocess 結果の配置先デバイス.
          pipeline builder がこの値に従って CPU / GPU preprocess 経路を切り替える.
    """
    device = config["device"]

    if is_tensorrt_model(model_path):
        return "cuda", "cuda"

    if is_onnx_model(model_path):
        active_providers = getattr(backend, "active_providers", [])
        actual_device = "cuda" if "CUDAExecutionProvider" in active_providers else "cpu"
        return actual_device, "cpu"

    return device, device
