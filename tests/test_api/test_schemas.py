"""DetectRequest / DetectionDict のバリデータを検証."""

import math

import pytest
from pydantic import ValidationError

from pochidetection.api.constants import MAX_PIXELS
from pochidetection.api.schemas import DetectionDict, DetectRequest


def test_detect_request_raw_requires_shape() -> None:
    """Raw 形式で shape 未指定はエラー."""
    with pytest.raises(ValidationError, match="shape が必須"):
        DetectRequest(image_data="", format="raw")


def test_detect_request_raw_rejects_invalid_shape_length() -> None:
    """Shape 長が 3 でないとエラー."""
    with pytest.raises(ValidationError, match=r"\[height, width, 3\]"):
        DetectRequest(image_data="", format="raw", shape=[480, 640])


def test_detect_request_jpeg_without_shape_ok() -> None:
    """JPEG 形式は shape 不要."""
    req = DetectRequest(image_data="AAA", format="jpeg")
    assert req.shape is None


def test_detect_request_rejects_non_uint8_dtype() -> None:
    """dtype は uint8 のみ許可."""
    with pytest.raises(ValidationError, match="許可されていない dtype"):
        DetectRequest(image_data="", format="raw", shape=[10, 10, 3], dtype="float32")


def test_detect_request_default_score_threshold() -> None:
    """デフォルト score_threshold は 0.5."""
    req = DetectRequest(image_data="", format="raw", shape=[10, 10, 3])
    assert req.score_threshold == 0.5


class TestDetectRequestScoreThresholdBounds:
    """score_threshold の境界値テスト. `ge=0.0, le=1.0`."""

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_accepts_in_range(self, value: float) -> None:
        """範囲内 (0.0, 境界, 1.0) は受理."""
        req = DetectRequest(
            image_data="", format="raw", shape=[10, 10, 3], score_threshold=value
        )
        assert req.score_threshold == value

    @pytest.mark.parametrize("value", [-0.01, -1.0, 1.01, 1.5, math.inf])
    def test_rejects_out_of_range(self, value: float) -> None:
        """範囲外 (負値 / 1 超 / inf) は ValidationError."""
        with pytest.raises(ValidationError):
            DetectRequest(
                image_data="",
                format="raw",
                shape=[10, 10, 3],
                score_threshold=value,
            )


class TestDetectRequestMaxPixels:
    """raw 形式の shape 上限 (MAX_PIXELS = 4096*4096) の境界値テスト."""

    def test_accepts_exact_max(self) -> None:
        """境界ちょうど (4096 x 4096) は受理."""
        req = DetectRequest(image_data="", format="raw", shape=[4096, 4096, 3])
        assert req.shape == [4096, 4096, 3]

    def test_rejects_one_pixel_over(self) -> None:
        """境界を 1 pixel 超過 (4097 x 4096) で ValidationError."""
        with pytest.raises(ValidationError, match="画像サイズが上限を超えています"):
            DetectRequest(image_data="", format="raw", shape=[4097, 4096, 3])

    def test_rejects_far_over_max(self) -> None:
        """大幅超過でも同じエラー."""
        # 8192 x 8192 = MAX_PIXELS の 4 倍
        with pytest.raises(ValidationError, match="画像サイズが上限を超えています"):
            DetectRequest(image_data="", format="raw", shape=[8192, 8192, 3])

    def test_max_pixels_constant(self) -> None:
        """MAX_PIXELS の値が 4096*4096 であることを確認 (ドキュメント整合)."""
        assert MAX_PIXELS == 4096 * 4096


def test_detection_dict_bbox_length_check() -> None:
    """Bbox 要素数が 4 でないとエラー."""
    with pytest.raises(ValidationError, match="4 要素"):
        DetectionDict(
            class_id=0, class_name="dog", confidence=0.9, bbox=[1.0, 2.0, 3.0]
        )


class TestDetectionDictConfidenceBounds:
    """DetectionDict.confidence の境界値テスト. `ge=0.0, le=1.0`."""

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_accepts_in_range(self, value: float) -> None:
        """範囲内は受理."""
        det = DetectionDict(
            class_id=0,
            class_name="dog",
            confidence=value,
            bbox=[1.0, 2.0, 3.0, 4.0],
        )
        assert det.confidence == value

    @pytest.mark.parametrize("value", [-0.01, -1.0, 1.01, 1.5])
    def test_rejects_out_of_range(self, value: float) -> None:
        """範囲外 (負値 / 1 超) は ValidationError."""
        with pytest.raises(ValidationError):
            DetectionDict(
                class_id=0,
                class_name="dog",
                confidence=value,
                bbox=[1.0, 2.0, 3.0, 4.0],
            )
