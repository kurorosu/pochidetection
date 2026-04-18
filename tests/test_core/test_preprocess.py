"""`pochidetection/core/preprocess.py` の単体テスト."""

import numpy as np
import torch

from pochidetection.core.preprocess import gpu_preprocess_tensor


def _sample_image(h: int = 10, w: int = 12) -> np.ndarray:
    """再現性のある RGB uint8 画像 (H, W, 3) を生成."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


class TestGpuPreprocessTensor:
    """gpu_preprocess_tensor の振る舞いを検証する."""

    def test_returns_float32_in_unit_interval(self) -> None:
        """fp16=False 時, pixel_values は float32 かつ [0, 1] に収まる."""
        image = _sample_image()
        target_hw = (8, 8)

        pixel_values, buf = gpu_preprocess_tensor(
            image, target_hw, device="cpu", input_buffer=None, use_fp16=False
        )

        assert pixel_values.shape == (1, 3, 8, 8)
        assert pixel_values.dtype == torch.float32
        assert buf.dtype == torch.float32
        assert pixel_values.min() >= 0.0
        assert pixel_values.max() <= 1.0

    def test_same_shape_reuses_buffer(self) -> None:
        """shape 不変なら第 2 戻り値 buffer が再利用される."""
        image = _sample_image()
        target_hw = (8, 8)

        _, buf1 = gpu_preprocess_tensor(
            image, target_hw, device="cpu", input_buffer=None, use_fp16=False
        )
        _, buf2 = gpu_preprocess_tensor(
            image, target_hw, device="cpu", input_buffer=buf1, use_fp16=False
        )

        assert buf1 is buf2

    def test_shape_mismatch_reallocates_buffer(self) -> None:
        """target_hw が変わると buffer が再確保される."""
        image = _sample_image()

        _, buf_small = gpu_preprocess_tensor(
            image, (8, 8), device="cpu", input_buffer=None, use_fp16=False
        )
        _, buf_large = gpu_preprocess_tensor(
            image, (16, 16), device="cpu", input_buffer=buf_small, use_fp16=False
        )

        assert buf_small is not buf_large
        assert buf_large.shape == (1, 3, 16, 16)

    def test_fp16_casts_pixel_values_but_preserves_buffer_dtype(self) -> None:
        """fp16=True 時, pixel_values は fp16 で, persisted buffer は float32 を維持."""
        image = _sample_image()
        target_hw = (8, 8)

        pixel_values, buf = gpu_preprocess_tensor(
            image, target_hw, device="cpu", input_buffer=None, use_fp16=True
        )

        assert pixel_values.dtype == torch.float16
        assert buf.dtype == torch.float32

    def test_fp16_buffer_survives_second_call(self) -> None:
        """fp16 経路を複数回呼んでも persisted buffer は float32 のまま."""
        image = _sample_image()
        target_hw = (8, 8)

        _, buf1 = gpu_preprocess_tensor(
            image, target_hw, device="cpu", input_buffer=None, use_fp16=True
        )
        _, buf2 = gpu_preprocess_tensor(
            image, target_hw, device="cpu", input_buffer=buf1, use_fp16=True
        )

        assert buf1 is buf2
        assert buf2.dtype == torch.float32

    def test_skips_resize_when_input_matches_target(self) -> None:
        """入力サイズ == target_hw の場合, resize を通さず値が保持される."""
        image = _sample_image(h=8, w=8)
        target_hw = (8, 8)

        pixel_values, _ = gpu_preprocess_tensor(
            image, target_hw, device="cpu", input_buffer=None, use_fp16=False
        )

        expected = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        assert torch.allclose(pixel_values, expected)
