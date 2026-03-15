"""ONNX Runtime Execution Providers の解決ユーティリティ."""

import onnxruntime as ort

_logger_configured: bool = False


def configure_ort_logger(severity: int = 3) -> None:
    """ONNX Runtime の C++ ロガーレベルを設定する.

    グローバル設定のため, 同一プロセス内で1回のみ実行する.
    RT-DETR の ScatterND オペレータが CUDA EP で大量の WARNING を出すため,
    デフォルトで ERROR (severity=3) 以上に制限する.

    Args:
        severity: ログレベル (0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL).
    """
    global _logger_configured  # noqa: PLW0603
    if _logger_configured:
        return
    ort.set_default_logger_severity(severity)
    _logger_configured = True


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
