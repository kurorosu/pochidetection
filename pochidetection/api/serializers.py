"""画像シリアライズ Strategy.

raw / jpeg 形式で base64 エンコードされた画像を numpy 配列に復元する.
deserialize の戻り値は BGR uint8 (cv2 convention). RGB 要求の推論パイプラインへは
呼び出し側 (backend) で変換する.

deserialize は ``(image, phase_times_ms)`` のタプルを返す. phase_times_ms には
b64_decode_ms などの計測値を含める.
"""

import base64
import time
from typing import Any, Protocol

import cv2
import numpy as np

from pochidetection.api.schemas import MAX_PIXELS


class IImageSerializer(Protocol):
    """画像シリアライズの Strategy インターフェース."""

    def serialize(self, image: np.ndarray) -> dict[str, Any]:
        """Numpy 配列をシリアライズ可能な辞書に変換する.

        Args:
            image: 画像配列 (H, W, 3) uint8 BGR.

        Returns:
            シリアライズされた辞書.
        """
        ...

    def deserialize(self, data: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
        """辞書から numpy 配列を復元する.

        Args:
            data: シリアライズされた辞書.

        Returns:
            (画像配列 (H, W, 3) uint8 BGR, フェーズ別所要時間 ms).
        """
        ...


class RawArraySerializer:
    """Raw numpy 配列の base64 エンコード. ローカル・開発用."""

    def serialize(self, image: np.ndarray) -> dict[str, Any]:
        """Numpy 配列を base64 エンコードする."""
        return {
            "image_data": base64.b64encode(image.tobytes()).decode("ascii"),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "format": "raw",
        }

    def deserialize(self, data: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
        """Base64 デコードして numpy 配列を復元する.

        Returns:
            (画像配列, {"b64_decode_ms": ..., "reshape_ms": ...}).

        Raises:
            ValueError: shape が未指定または不正な場合.
        """
        if "shape" not in data or data["shape"] is None:
            raise ValueError("raw フォーマットでは shape が必須です")

        shape = tuple(data["shape"])
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(
                f"shape は (H, W, 3) である必要があります. 受け取った: {shape}"
            )

        height, width = shape[0], shape[1]
        if height * width > MAX_PIXELS:
            raise ValueError(
                f"画像サイズが上限を超えています: {height}x{width} "
                f"(上限: {MAX_PIXELS} ピクセル)"
            )

        t0 = time.perf_counter()
        raw_bytes = base64.b64decode(data["image_data"])
        t1 = time.perf_counter()
        dtype = np.dtype(data.get("dtype", "uint8"))
        image = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
        t2 = time.perf_counter()

        phase_times = {
            "b64_decode_ms": (t1 - t0) * 1000,
            "reshape_ms": (t2 - t1) * 1000,
        }
        return image, phase_times


class JpegSerializer:
    """JPEG 圧縮. ネットワーク転送用."""

    def __init__(self, quality: int = 90) -> None:
        """Initialize with JPEG quality.

        Args:
            quality: JPEG 品質 (1-100).
        """
        self.quality = quality

    def serialize(self, image: np.ndarray) -> dict[str, Any]:
        """Encode the image as JPEG and return a base64 dict."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        _, buf = cv2.imencode(".jpg", image, encode_params)
        return {
            "image_data": base64.b64encode(buf.tobytes()).decode("ascii"),
            "format": "jpeg",
        }

    def deserialize(self, data: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
        """Decode base64 JPEG into a numpy BGR array.

        Returns:
            (画像配列 BGR, {"b64_decode_ms": ..., "imdecode_ms": ...}).

        Raises:
            ValueError: JPEG デコードに失敗した場合.
        """
        t0 = time.perf_counter()
        jpeg_bytes = base64.b64decode(data["image_data"])
        t1 = time.perf_counter()
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        t2 = time.perf_counter()
        if image is None:
            raise ValueError("JPEG デコード失敗: 不正または破損した画像データ")
        phase_times = {
            "b64_decode_ms": (t1 - t0) * 1000,
            "imdecode_ms": (t2 - t1) * 1000,
        }
        return image, phase_times


def create_serializer(fmt: str) -> IImageSerializer:
    """Create a serializer for the given format.

    Args:
        fmt: 形式 ("raw" or "jpeg").

    Returns:
        シリアライザインスタンス.

    Raises:
        ValueError: サポートされていない形式の場合.
    """
    if fmt == "raw":
        return RawArraySerializer()
    if fmt == "jpeg":
        return JpegSerializer()
    raise ValueError(
        f"サポートされていない形式: {fmt}. 'raw' or 'jpeg' を指定してください"
    )
