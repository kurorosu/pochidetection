"""Letterbox リサイズの共通ユーティリティ.

アスペクト比を保ったまま target サイズに収め, 余白を padding で埋める
前処理 (letterbox) を backend-agnostic に提供する.

学習側 (``datasets/transforms/letterbox.py::LetterboxTransform``) と
推論側 (``core/preprocess.py::gpu_preprocess_tensor`` 予定, Issue #445) の
両方から import して使い回せるよう, 幾何計算 (``compute_letterbox_params``) と
ピクセル操作 (``apply_letterbox``) を純粋関数として切り出している.
"""

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.transforms import v2


@dataclass(frozen=True)
class LetterboxParams:
    """Letterbox 変換の幾何パラメータ.

    Attributes:
        scale: 元画像に掛ける拡大縮小率 (``min(dst_h/src_h, dst_w/src_w)``).
        new_h: scale 適用後の高さ (``round(src_h * scale)``).
        new_w: scale 適用後の幅 (``round(src_w * scale)``).
        pad_top: 上側 padding ピクセル数.
        pad_left: 左側 padding ピクセル数.
        pad_bottom: 下側 padding ピクセル数.
        pad_right: 右側 padding ピクセル数.
    """

    scale: float
    new_h: int
    new_w: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int


def compute_letterbox_params(
    src_hw: tuple[int, int], dst_hw: tuple[int, int]
) -> LetterboxParams:
    """元画像サイズと target サイズから letterbox 幾何パラメータを計算する.

    Args:
        src_hw: 元画像の (height, width).
        dst_hw: target の (height, width).

    Returns:
        計算済みの ``LetterboxParams``. scale は長辺が target に収まる倍率.
        pad は余り方向 (短辺側) に両側均等分配し, 奇数差は下 / 右側に 1 多く配分.

    Raises:
        ValueError: src_hw / dst_hw いずれかが正値でない場合.
    """
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError(
            f"src_hw と dst_hw は正の整数タプルである必要があります: "
            f"src={src_hw}, dst={dst_hw}"
        )

    scale = min(dst_h / src_h, dst_w / src_w)
    new_h = round(src_h * scale)
    new_w = round(src_w * scale)

    pad_vertical = dst_h - new_h
    pad_horizontal = dst_w - new_w
    pad_top = pad_vertical // 2
    pad_left = pad_horizontal // 2
    pad_bottom = pad_vertical - pad_top
    pad_right = pad_horizontal - pad_left

    return LetterboxParams(
        scale=scale,
        new_h=new_h,
        new_w=new_w,
        pad_top=pad_top,
        pad_left=pad_left,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
    )


def apply_letterbox(
    image: Image.Image | torch.Tensor,
    params: LetterboxParams,
    pad_value: int = 0,
) -> Image.Image | torch.Tensor:
    """Letterbox を画像に適用する (PIL.Image / torch.Tensor 多態).

    ``v2.functional.resize`` で scale 適用後サイズにリサイズし,
    ``v2.functional.pad`` で target サイズに padding する.

    Args:
        image: 入力画像. ``PIL.Image.Image`` または ``torch.Tensor`` (C,H,W).
        params: ``compute_letterbox_params`` の戻り値.
        pad_value: padding に使うピクセル値 (default ``0``).

    Returns:
        入力と同じ型の画像. サイズは ``(new_h + pad_vertical, new_w + pad_horizontal)``
        = target_hw.
    """
    resized = v2.functional.resize(
        image,
        [params.new_h, params.new_w],
        interpolation=v2.InterpolationMode.BILINEAR,
        antialias=True,
    )
    # v2.functional.pad の padding は [left, top, right, bottom] 順.
    return v2.functional.pad(
        resized,
        [params.pad_left, params.pad_top, params.pad_right, params.pad_bottom],
        fill=pad_value,
    )
