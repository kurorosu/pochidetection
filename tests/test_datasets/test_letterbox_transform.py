"""`LetterboxTransform` (v2.Transform) の bbox 同期変換テスト."""

import pytest
import torch
from PIL import Image
from torchvision import tv_tensors

from pochidetection.datasets.transforms import LetterboxTransform


def _make_image(h: int, w: int, color: int = 128) -> Image.Image:
    """単色 RGB PIL 画像を生成."""
    return Image.new("RGB", (w, h), color=(color, color, color))


def _xyxy_bboxes(
    boxes: list[list[float]], canvas_hw: tuple[int, int]
) -> tv_tensors.BoundingBoxes:
    """XYXY 形式の BoundingBoxes を生成."""
    return tv_tensors.BoundingBoxes(
        boxes,
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=canvas_hw,
    )


class TestLetterboxTransformInit:
    """``__init__`` のバリデーション."""

    def test_int_size_is_square(self) -> None:
        """int 指定は正方に展開される."""
        t = LetterboxTransform(640)
        assert t.size == (640, 640)

    def test_tuple_size_is_hw(self) -> None:
        """タプル指定は (H, W) として保持."""
        t = LetterboxTransform((480, 640))
        assert t.size == (480, 640)

    def test_invalid_length_raises(self) -> None:
        """長さ 2 以外は ValueError."""
        with pytest.raises(ValueError, match="長さ 2"):
            LetterboxTransform([640, 640, 3])

    def test_non_positive_size_raises(self) -> None:
        """非正値は ValueError."""
        with pytest.raises(ValueError, match="正の整数"):
            LetterboxTransform((0, 640))


class TestLetterboxTransformSquare:
    """正方入力 → 正方 target (no-op 相当)."""

    def test_image_and_bbox_preserved(self) -> None:
        """scale=1, pad=0 で画像サイズも bbox も維持される."""
        image = _make_image(640, 640)
        bboxes = _xyxy_bboxes([[100.0, 100.0, 200.0, 200.0]], (640, 640))

        transform = LetterboxTransform(640)
        out_image, out_bboxes = transform(image, bboxes)

        assert out_image.size == (640, 640)
        assert isinstance(out_bboxes, tv_tensors.BoundingBoxes)
        assert out_bboxes.canvas_size == (640, 640)
        # bbox coords 不変
        torch.testing.assert_close(
            out_bboxes.as_subclass(torch.Tensor),
            torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
        )


class TestLetterboxTransformLandscape:
    """横長入力 (1280x720) → 正方 640 (上下に 140 pad)."""

    def test_image_size_matches_target(self) -> None:
        """画像サイズが target に揃う."""
        image = _make_image(720, 1280)
        transform = LetterboxTransform(640)

        out_image = transform(image)
        assert out_image.size == (640, 640)  # PIL は (W, H)

    def test_bbox_scaled_and_pad_top_added(self) -> None:
        """bbox は x 軸 scale 0.5, y 軸 scale 0.5 + pad_top=140 加算される."""
        image = _make_image(720, 1280)
        # 元座標 XYXY: [200, 100, 600, 300] (元画像 1280x720 基準)
        bboxes = _xyxy_bboxes([[200.0, 100.0, 600.0, 300.0]], (720, 1280))

        transform = LetterboxTransform(640)
        _, out_bboxes = transform(image, bboxes)

        assert out_bboxes.canvas_size == (640, 640)
        # scale = 0.5, pad_top = 140, pad_left = 0
        # 期待: [200*0.5 + 0, 100*0.5 + 140, 600*0.5 + 0, 300*0.5 + 140]
        #     = [100, 190, 300, 290]
        torch.testing.assert_close(
            out_bboxes.as_subclass(torch.Tensor),
            torch.tensor([[100.0, 190.0, 300.0, 290.0]]),
        )


class TestLetterboxTransformPortrait:
    """縦長入力 (720x1280) → 正方 640 (左右に 140 pad)."""

    def test_image_size_matches_target(self) -> None:
        """画像サイズが target に揃う."""
        image = _make_image(1280, 720)
        transform = LetterboxTransform(640)

        out_image = transform(image)
        assert out_image.size == (640, 640)

    def test_bbox_scaled_and_pad_left_added(self) -> None:
        """bbox は scale 0.5 + pad_left=140 加算される."""
        image = _make_image(1280, 720)
        bboxes = _xyxy_bboxes([[100.0, 200.0, 300.0, 600.0]], (1280, 720))

        transform = LetterboxTransform(640)
        _, out_bboxes = transform(image, bboxes)

        assert out_bboxes.canvas_size == (640, 640)
        # scale = 0.5, pad_top = 0, pad_left = 140
        # 期待: [100*0.5 + 140, 200*0.5 + 0, 300*0.5 + 140, 600*0.5 + 0]
        #     = [190, 100, 290, 300]
        torch.testing.assert_close(
            out_bboxes.as_subclass(torch.Tensor),
            torch.tensor([[190.0, 100.0, 290.0, 300.0]]),
        )


class TestLetterboxTransformTensorImage:
    """tv_tensors.Image (Tensor) 入力も正しく動作する."""

    def test_tensor_image_landscape(self) -> None:
        """Tensor 画像 (C,H,W) でも target サイズに揃う."""
        image = tv_tensors.Image(torch.zeros((3, 720, 1280), dtype=torch.uint8))
        transform = LetterboxTransform(640)
        out = transform(image)

        assert out.shape == (3, 640, 640)
