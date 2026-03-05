"""デバイス関連ユーティリティ."""


def is_fp16_available(use_fp16: bool, device: str) -> bool:
    """FP16 推論が利用可能かを判定する.

    FP16 は CUDA デバイスでのみ有効.

    Args:
        use_fp16: FP16 使用フラグ.
        device: デバイス文字列 (例: "cuda", "cpu").

    Returns:
        FP16 が利用可能なら True.
    """
    return use_fp16 and device == "cuda"
