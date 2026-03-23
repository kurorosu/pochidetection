"""GPU/CPU リソース使用状況の取得."""

from dataclasses import dataclass

import psutil
import torch


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """リソース使用状況のスナップショット.

    Attributes:
        gpu_utilization: GPU 使用率 (%). CUDA 非利用時は None.
        vram_used_gb: GPU メモリ使用量 (GB). CUDA 非利用時は None.
        vram_total_gb: GPU メモリ総量 (GB). CUDA 非利用時は None.
        cpu_percent: CPU 使用率 (%).
    """

    gpu_utilization: int | None
    vram_used_gb: float | None
    vram_total_gb: float | None
    cpu_percent: float


def get_resource_usage() -> ResourceUsage:
    """現在のリソース使用状況を取得する.

    Returns:
        リソース使用状況のスナップショット.
    """
    cpu_percent = psutil.cpu_percent(interval=None)

    gpu_utilization = None
    vram_used_gb = None
    vram_total_gb = None

    if torch.cuda.is_available():
        try:
            gpu_utilization = torch.cuda.utilization()
            free, total = torch.cuda.mem_get_info()
            used = total - free
            vram_used_gb = round(used / (1024**3), 1)
            vram_total_gb = round(total / (1024**3), 1)
        except (RuntimeError, ModuleNotFoundError):
            # Why: pynvml 未インストール or CUDA デバイスなし環境で発生.
            pass

    return ResourceUsage(
        gpu_utilization=gpu_utilization,
        vram_used_gb=vram_used_gb,
        vram_total_gb=vram_total_gb,
        cpu_percent=cpu_percent,
    )


def format_resource_lines(usage: ResourceUsage) -> list[str]:
    """リソース使用状況をオーバーレイ用の文字列リストに変換する.

    Args:
        usage: リソース使用状況.

    Returns:
        描画用テキスト行のリスト.
    """
    lines: list[str] = []
    if usage.gpu_utilization is not None:
        lines.append(f"GPU:  {usage.gpu_utilization}%")
    if usage.vram_used_gb is not None and usage.vram_total_gb is not None:
        lines.append(f"VRAM: {usage.vram_used_gb}/{usage.vram_total_gb}GB")
    lines.append(f"CPU:  {usage.cpu_percent:.0f}%")
    return lines
