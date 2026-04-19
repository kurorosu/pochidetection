"""`pochidetection.core.letterbox` のテスト."""

import numpy as np
import pytest
import torch
from PIL import Image

from pochidetection.core.letterbox import (
    LetterboxParams,
    apply_letterbox,
    compute_letterbox_params,
)


class TestComputeLetterboxParams:
    """``compute_letterbox_params`` の幾何計算を検証."""

    def test_square_source_is_noop_like(self) -> None:
        """正方画像 → 正方 target は scale=1, pad=0."""
        params = compute_letterbox_params((640, 640), (640, 640))
        assert params == LetterboxParams(
            scale=1.0,
            new_h=640,
            new_w=640,
            pad_top=0,
            pad_left=0,
            pad_bottom=0,
            pad_right=0,
        )

    def test_landscape_source_pads_vertically(self) -> None:
        """横長 (1280x720) → 正方 640 は幅がピッタリで上下に 140 ずつ pad."""
        params = compute_letterbox_params((720, 1280), (640, 640))
        assert params.scale == pytest.approx(0.5)
        assert (params.new_h, params.new_w) == (360, 640)
        # pad_vertical = 640 - 360 = 280 → 上下 140 ずつ
        assert (params.pad_top, params.pad_bottom) == (140, 140)
        assert (params.pad_left, params.pad_right) == (0, 0)

    def test_portrait_source_pads_horizontally(self) -> None:
        """縦長 (720x1280) → 正方 640 は高さがピッタリで左右に 140 ずつ pad."""
        params = compute_letterbox_params((1280, 720), (640, 640))
        assert params.scale == pytest.approx(0.5)
        assert (params.new_h, params.new_w) == (640, 360)
        assert (params.pad_top, params.pad_bottom) == (0, 0)
        # pad_horizontal = 640 - 360 = 280 → 左右 140 ずつ
        assert (params.pad_left, params.pad_right) == (140, 140)

    def test_odd_pad_difference_goes_to_bottom_right(self) -> None:
        """奇数 pad 差分は下 / 右側に 1 多く配分される."""
        # 正方 100 → 101 : pad_vertical = pad_horizontal = 101 - 100 = 1
        # ...ではなく, 100x101 → 101x101 だと scale=1.0 で new=100x101, pad は (1,0)
        params = compute_letterbox_params((100, 101), (101, 101))
        assert params.scale == pytest.approx(1.0)
        assert (params.new_h, params.new_w) == (100, 101)
        # pad_vertical=1 → top=0, bottom=1
        assert (params.pad_top, params.pad_bottom) == (0, 1)
        assert (params.pad_left, params.pad_right) == (0, 0)

    def test_invalid_src_hw_raises(self) -> None:
        """非正値は ValueError."""
        with pytest.raises(ValueError, match="正の整数タプル"):
            compute_letterbox_params((0, 640), (640, 640))

    def test_invalid_dst_hw_raises(self) -> None:
        """非正値は ValueError."""
        with pytest.raises(ValueError, match="正の整数タプル"):
            compute_letterbox_params((640, 640), (640, -1))


class TestApplyLetterboxPil:
    """``apply_letterbox`` に ``PIL.Image`` を渡した経路."""

    def test_pil_landscape_output_size_matches_target(self) -> None:
        """PIL 入力 → PIL 出力. サイズが target に揃う."""
        src = Image.new("RGB", (1280, 720), color=(200, 150, 100))
        params = compute_letterbox_params((720, 1280), (640, 640))

        out = apply_letterbox(src, params, pad_value=0)

        assert isinstance(out, Image.Image)
        # PIL.Image.size は (width, height)
        assert out.size == (640, 640)

    def test_pil_padding_region_is_zero(self) -> None:
        """横長入力時, padding 領域 (上下の端) は pad_value=0."""
        # 単色 200 の PIL 画像で letterbox 適用後, 上下端の行が 0 であることを確認
        src = Image.new("RGB", (1280, 720), color=(200, 200, 200))
        params = compute_letterbox_params((720, 1280), (640, 640))

        out = apply_letterbox(src, params, pad_value=0)
        out_np = np.array(out)

        # 上端の 1 行目 (pad_top 領域内) は 0
        assert np.all(out_np[0] == 0)
        # 下端の最終行 (pad_bottom 領域内) は 0
        assert np.all(out_np[-1] == 0)
        # 中央付近 (new_h の真ん中) は元色 200 が残っている
        assert np.all(out_np[320] > 0)


class TestApplyLetterboxTensor:
    """``apply_letterbox`` に ``torch.Tensor`` を渡した経路 (#445 GPU 経路の再利用保証)."""

    def test_tensor_landscape_output_shape_matches_target(self) -> None:
        """Tensor (C, H, W) 入力 → Tensor 出力. 形状が target に揃う."""
        src = torch.full((3, 720, 1280), 128, dtype=torch.uint8)
        params = compute_letterbox_params((720, 1280), (640, 640))

        out = apply_letterbox(src, params, pad_value=0)

        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 640, 640)
        assert out.dtype == torch.uint8

    def test_tensor_padding_region_is_zero(self) -> None:
        """Tensor 入力時, 上下 padding 領域が 0."""
        src = torch.full((3, 720, 1280), 200, dtype=torch.uint8)
        params = compute_letterbox_params((720, 1280), (640, 640))

        out = apply_letterbox(src, params, pad_value=0)

        # 上端 (pad_top = 140 ピクセル) は全て 0
        assert torch.all(out[:, : params.pad_top, :] == 0)
        # 下端 (pad_bottom = 140 ピクセル) は全て 0
        assert torch.all(out[:, -params.pad_bottom :, :] == 0)
        # 中央の 1 行は元値 200 に近い (BILINEAR 補間なので厳密一致しない)
        center_row = out[:, params.pad_top + params.new_h // 2, :]
        assert torch.all(center_row > 0)

    def test_tensor_batched_input_also_works(self) -> None:
        """Batch 次元 (N, C, H, W) 入力も v2.functional 経由で動く."""
        src = torch.full((1, 3, 720, 1280), 100, dtype=torch.uint8)
        params = compute_letterbox_params((720, 1280), (640, 640))

        out = apply_letterbox(src, params, pad_value=0)

        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 3, 640, 640)

    def test_square_source_is_near_noop(self) -> None:
        """正方 src → 正方 dst で同サイズの場合は scale=1, pad=0 の no-op 相当."""
        src = torch.full((3, 640, 640), 100, dtype=torch.uint8)
        params = compute_letterbox_params((640, 640), (640, 640))

        out = apply_letterbox(src, params, pad_value=0)

        assert out.shape == (3, 640, 640)
        # 全ピクセルが元値 100 (resize も pad も実質 no-op)
        assert torch.all(out == 100)
