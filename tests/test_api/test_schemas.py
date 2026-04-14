"""DetectRequest / DetectionDict のバリデータを検証."""

import pytest
from pydantic import ValidationError

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


def test_detect_request_score_threshold_bounds() -> None:
    """score_threshold は 0-1 の範囲."""
    with pytest.raises(ValidationError):
        DetectRequest(
            image_data="", format="raw", shape=[10, 10, 3], score_threshold=1.5
        )


def test_detect_request_default_score_threshold() -> None:
    """デフォルト score_threshold は 0.5."""
    req = DetectRequest(image_data="", format="raw", shape=[10, 10, 3])
    assert req.score_threshold == 0.5


def test_detection_dict_bbox_length_check() -> None:
    """Bbox 要素数が 4 でないとエラー."""
    with pytest.raises(ValidationError, match="4 要素"):
        DetectionDict(
            class_id=0, class_name="dog", confidence=0.9, bbox=[1.0, 2.0, 3.0]
        )
