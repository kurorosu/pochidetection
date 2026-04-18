"""推論パイプライン共通の前処理ヘルパー."""

import numpy as np
import torch
from torchvision.transforms import v2


def gpu_preprocess_tensor(
    image_np: np.ndarray,
    target_hw: tuple[int, int],
    device: torch.device | str,
    input_buffer: torch.Tensor | None,
    use_fp16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU 経路の画像前処理を実行する.

    numpy RGB uint8 → uint8 tensor CHW → CPU 上で uint8 のまま resize →
    GPU バッファに ``copy_`` で float32 化 + H2D → ``div_(255)`` で [0,1] 化.
    バッファは shape mismatch 時のみ再確保する.

    Why:
        ``copy_`` による uint8→float32 キャストと H2D 転送の統合で
        CUDA コピー回数を削減する. persisted buffer は常に float32 で保持し,
        fp16 経路でも dtype を変えない (次回 ``copy_(uint8)`` 時の事故を防ぐ).

    Args:
        image_np: RGB uint8 numpy 配列 (H, W, 3).
        target_hw: リサイズ先の (height, width).
        device: GPU バッファの配置先.
        input_buffer: 前回呼び出しで確保した float32 バッファ. None または
            shape mismatch 時は再確保する.
        use_fp16: True の場合, 戻り値 ``pixel_values`` を fp16 にキャストする.
            persisted buffer は float32 のまま維持する.

    Returns:
        (pixel_values, persisted_buffer) のタプル.
            - pixel_values: 推論入力用テンソル (1, C, H, W).
              ``use_fp16=True`` なら fp16, そうでなければ persisted_buffer と同一参照.
            - persisted_buffer: 次回呼び出しで再利用するための float32 バッファ.
    """
    tensor_uint8 = torch.from_numpy(image_np).permute(2, 0, 1)  # (C, H, W) uint8
    target_h, target_w = target_hw
    if tensor_uint8.shape[1:] != (target_h, target_w):
        tensor_uint8 = v2.functional.resize(
            tensor_uint8,
            [target_h, target_w],
            interpolation=v2.InterpolationMode.BILINEAR,
        )
    tensor_uint8 = tensor_uint8.unsqueeze(0)  # (1, C, H, W) uint8

    buf = input_buffer
    if buf is None or buf.shape != tensor_uint8.shape:
        buf = torch.empty(tensor_uint8.shape, dtype=torch.float32, device=device)
    # Why: copy_ は dtype 変換兼 H2D. float32 buffer に uint8 を入れることで
    # uint8→float32 キャストと CPU→GPU 転送を 1 度の CUDA コピーで済ませる.
    buf.copy_(tensor_uint8)
    buf.div_(255.0)

    pixel_values = buf.half() if use_fp16 else buf
    return pixel_values, buf
