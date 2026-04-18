"""画像シリアライザの round-trip とエラーケースを検証."""

import base64

import numpy as np
import pytest

from pochidetection.api.serializers import (
    JpegSerializer,
    RawArraySerializer,
    create_serializer,
)


def test_raw_serializer_round_trip() -> None:
    """RawArraySerializer は serialize→deserialize で同一配列を復元する."""
    image = np.random.randint(0, 255, size=(32, 48, 3), dtype=np.uint8)
    serializer = RawArraySerializer()
    data = serializer.serialize(image)
    restored = serializer.deserialize(data)
    np.testing.assert_array_equal(restored, image)


def test_raw_serializer_rejects_missing_shape() -> None:
    """RawArraySerializer は shape 未指定でエラー."""
    serializer = RawArraySerializer()
    with pytest.raises(ValueError, match="shape が必須"):
        serializer.deserialize({"image_data": "", "shape": None})


def test_raw_serializer_rejects_invalid_channel() -> None:
    """RawArraySerializer は channel != 3 でエラー."""
    serializer = RawArraySerializer()
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        serializer.deserialize({"image_data": "", "shape": [10, 10, 4]})


def test_jpeg_serializer_preserves_shape() -> None:
    """JpegSerializer は shape と dtype を保つ (値は非可逆)."""
    image = (np.ones((16, 24, 3), dtype=np.uint8) * 128).astype(np.uint8)
    serializer = JpegSerializer()
    data = serializer.serialize(image)
    restored = serializer.deserialize(data)
    assert restored.shape == image.shape
    assert restored.dtype == np.uint8


def test_jpeg_serializer_rejects_invalid_bytes() -> None:
    """JpegSerializer は不正バイト列でエラー."""
    serializer = JpegSerializer()
    bad = base64.b64encode(b"not a jpeg").decode("ascii")
    with pytest.raises(ValueError, match="JPEG デコード失敗"):
        serializer.deserialize({"image_data": bad})


def test_create_serializer_raw() -> None:
    assert isinstance(create_serializer("raw"), RawArraySerializer)


def test_create_serializer_jpeg() -> None:
    assert isinstance(create_serializer("jpeg"), JpegSerializer)


def test_create_serializer_unknown() -> None:
    with pytest.raises(ValueError, match="サポートされていない形式"):
        create_serializer("png")
