"""`pochidetection/core/preprocess.py` の単体テスト."""

import numpy as np
import pytest
import torch

from pochidetection.core.letterbox import compute_letterbox_params
from pochidetection.core.preprocess import gpu_preprocess_tensor

_DEVICE_PARAMS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
        ),
    ),
]


def _sample_image(h: int = 10, w: int = 12) -> np.ndarray:
    """再現性のある RGB uint8 画像 (H, W, 3) を生成."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


class TestGpuPreprocessTensor:
    """gpu_preprocess_tensor の振る舞いを検証する. cpu / cuda 両 device で実機検証."""

    @pytest.mark.parametrize("device", _DEVICE_PARAMS)
    def test_returns_float32_in_unit_interval(self, device: str) -> None:
        """fp16=False 時, pixel_values は float32 かつ [0, 1] に収まる."""
        image = _sample_image()
        target_hw = (8, 8)

        pixel_values, buf = gpu_preprocess_tensor(
            image, target_hw, device=device, input_buffer=None, use_fp16=False
        )

        assert pixel_values.shape == (1, 3, 8, 8)
        assert pixel_values.dtype == torch.float32
        assert buf.dtype == torch.float32
        assert str(pixel_values.device).startswith(device)
        assert str(buf.device).startswith(device)
        assert pixel_values.min() >= 0.0
        assert pixel_values.max() <= 1.0

    @pytest.mark.parametrize("device", _DEVICE_PARAMS)
    def test_same_shape_reuses_buffer(self, device: str) -> None:
        """shape 不変なら第 2 戻り値 buffer が再利用される."""
        image = _sample_image()
        target_hw = (8, 8)

        _, buf1 = gpu_preprocess_tensor(
            image, target_hw, device=device, input_buffer=None, use_fp16=False
        )
        _, buf2 = gpu_preprocess_tensor(
            image, target_hw, device=device, input_buffer=buf1, use_fp16=False
        )

        assert buf1 is buf2

    @pytest.mark.parametrize("device", _DEVICE_PARAMS)
    def test_shape_mismatch_reallocates_buffer(self, device: str) -> None:
        """target_hw が変わると buffer が再確保される."""
        image = _sample_image()

        _, buf_small = gpu_preprocess_tensor(
            image, (8, 8), device=device, input_buffer=None, use_fp16=False
        )
        _, buf_large = gpu_preprocess_tensor(
            image, (16, 16), device=device, input_buffer=buf_small, use_fp16=False
        )

        assert buf_small is not buf_large
        assert buf_large.shape == (1, 3, 16, 16)
        assert str(buf_large.device).startswith(device)

    @pytest.mark.parametrize("device", _DEVICE_PARAMS)
    def test_fp16_casts_pixel_values_but_preserves_buffer_dtype(
        self, device: str
    ) -> None:
        """fp16=True 時, pixel_values は fp16 で, persisted buffer は float32 を維持."""
        image = _sample_image()
        target_hw = (8, 8)

        pixel_values, buf = gpu_preprocess_tensor(
            image, target_hw, device=device, input_buffer=None, use_fp16=True
        )

        assert pixel_values.dtype == torch.float16
        assert buf.dtype == torch.float32
        assert str(pixel_values.device).startswith(device)

    @pytest.mark.parametrize("device", _DEVICE_PARAMS)
    def test_fp16_buffer_survives_second_call(self, device: str) -> None:
        """fp16 経路を複数回呼んでも persisted buffer は float32 のまま."""
        image = _sample_image()
        target_hw = (8, 8)

        _, buf1 = gpu_preprocess_tensor(
            image, target_hw, device=device, input_buffer=None, use_fp16=True
        )
        _, buf2 = gpu_preprocess_tensor(
            image, target_hw, device=device, input_buffer=buf1, use_fp16=True
        )

        assert buf1 is buf2
        assert buf2.dtype == torch.float32

    @pytest.mark.parametrize("device", _DEVICE_PARAMS)
    def test_skips_resize_when_input_matches_target(self, device: str) -> None:
        """入力サイズ == target_hw の場合, resize を通さず値が保持される."""
        image = _sample_image(h=8, w=8)
        target_hw = (8, 8)

        pixel_values, _ = gpu_preprocess_tensor(
            image, target_hw, device=device, input_buffer=None, use_fp16=False
        )

        expected = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(pixel_values.device)
            / 255.0
        )
        assert torch.allclose(pixel_values, expected)


class TestGpuPreprocessTensorLetterbox:
    """``letterbox_params`` オプションが letterbox 経路で動作することを検証."""

    @pytest.mark.parametrize("device", _DEVICE_PARAMS)
    def test_letterbox_params_applies_padding(self, device: str) -> None:
        """横長入力 + letterbox_params 指定で上下 pad が 0 (normalize 後も 0)."""
        # 横長 (1, 2) → (2, 2): scale=1.0, new=(1,2), pad_top=0,pad_bottom=1
        # ではなく (2, 4) → (4, 4): scale=1.0, new=(2,4), pad_top=1,pad_bottom=1
        # 実テストは (4, 8) → (8, 8) を使う
        image = _sample_image(h=4, w=8)
        target_hw = (8, 8)
        params = compute_letterbox_params((4, 8), target_hw)

        assert params.new_h == 4
        assert params.new_w == 8
        assert params.pad_top == 2
        assert params.pad_bottom == 2

        pixel_values, _ = gpu_preprocess_tensor(
            image,
            target_hw,
            device=device,
            input_buffer=None,
            use_fp16=False,
            letterbox_params=params,
        )

        assert pixel_values.shape == (1, 3, 8, 8)
        # 上端 pad_top (2 行) と下端 pad_bottom (2 行) が 0.
        assert torch.all(pixel_values[0, :, :2, :] == 0)
        assert torch.all(pixel_values[0, :, -2:, :] == 0)

    @pytest.mark.parametrize("device", _DEVICE_PARAMS)
    def test_letterbox_params_none_falls_back_to_resize(self, device: str) -> None:
        """letterbox_params=None は従来 resize 経路と同等 (後方互換)."""
        image = _sample_image(h=10, w=12)
        target_hw = (8, 8)

        pv_default, _ = gpu_preprocess_tensor(
            image,
            target_hw,
            device=device,
            input_buffer=None,
            use_fp16=False,
        )
        pv_explicit_none, _ = gpu_preprocess_tensor(
            image,
            target_hw,
            device=device,
            input_buffer=None,
            use_fp16=False,
            letterbox_params=None,
        )

        assert torch.allclose(pv_default, pv_explicit_none)
