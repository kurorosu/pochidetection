"""ONNX Runtime Execution Providers の解決ユーティリティ."""

import onnxruntime as ort


def resolve_providers(device: str) -> list[str]:
    """デバイス設定に応じた Execution Providers を返す.

    Args:
        device: 推論デバイス ("cpu" または "cuda").

    Returns:
        Execution Providers のリスト.
    """
    if device == "cuda":
        available = ort.get_available_providers()
        providers: list[str] = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    return ["CPUExecutionProvider"]
