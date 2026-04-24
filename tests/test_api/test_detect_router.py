"""POST /api/v1/detect エンドポイントを MagicMock engine で検証."""

import base64
from collections.abc import Iterator
from unittest.mock import patch

import numpy as np
import pynvml
import pytest
from fastapi.testclient import TestClient

from pochidetection.api import gpu_metrics
from pochidetection.api.app import create_app
from pochidetection.api.state import set_engine


@pytest.fixture(autouse=True)
def _reset_gpu_metrics_state() -> Iterator[None]:
    """`gpu_metrics` モジュール level のキャッシュ状態をテスト間で初期化する."""
    gpu_metrics._handle = None
    gpu_metrics._initialized = False
    yield
    gpu_metrics._handle = None
    gpu_metrics._initialized = False


def _encode_raw(image: np.ndarray) -> dict[str, object]:
    return {
        "image_data": base64.b64encode(image.tobytes()).decode("ascii"),
        "format": "raw",
        "shape": list(image.shape),
        "dtype": "uint8",
    }


def _install_engine_returning(detections: list[dict]) -> None:
    from unittest.mock import MagicMock

    engine = MagicMock()
    engine.backend_name = "pytorch"
    engine.predict.return_value = (detections, {})
    set_engine(engine)


def test_detect_returns_200_with_detections() -> None:
    """Engine が返した検出をそのままレスポンスに詰める."""
    _install_engine_returning(
        [
            {
                "class_id": 0,
                "class_name": "dog",
                "confidence": 0.9,
                "bbox": [1.0, 2.0, 3.0, 4.0],
            }
        ]
    )
    app = create_app(None)
    payload = _encode_raw(np.zeros((4, 4, 3), dtype=np.uint8))
    with TestClient(app) as client:
        res = client.post("/api/v1/detect", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["backend"] == "pytorch"
    assert body["detections"][0]["class_name"] == "dog"
    assert body["detections"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert "e2e_time_ms" in body
    assert "phase_times_ms" in body
    # MagicMock engine が空 dict を返すため breakdown は出ない.
    # 旧 b64_decode_ms / cvt_color_ms 等のキーが消えていることを確認.
    assert "b64_decode_ms" not in body["phase_times_ms"]
    assert "cvt_color_ms" not in body["phase_times_ms"]
    assert "gap_since_last_request_ms" not in body["phase_times_ms"]


def test_detect_returns_503_when_engine_missing() -> None:
    """Engine 未初期化時は 503."""
    app = create_app(None)
    payload = _encode_raw(np.zeros((4, 4, 3), dtype=np.uint8))
    with TestClient(app) as client:
        res = client.post("/api/v1/detect", json=payload)
    assert res.status_code == 503


def test_detect_returns_400_on_invalid_base64() -> None:
    """不正な base64 (binascii.Error) は 400 で返す."""
    _install_engine_returning([])
    app = create_app(None)
    payload = {
        "image_data": "!!!not-base64!!!",
        "format": "raw",
        "shape": [4, 4, 3],
        "dtype": "uint8",
    }
    with TestClient(app) as client:
        res = client.post("/api/v1/detect", json=payload)
    assert res.status_code == 400


def test_detect_returns_422_when_shape_missing() -> None:
    """Raw 形式で shape 欠落は 422 (Pydantic バリデーション)."""
    _install_engine_returning([])
    app = create_app(None)
    with TestClient(app) as client:
        res = client.post(
            "/api/v1/detect",
            json={"image_data": "AAA", "format": "raw"},
        )
    assert res.status_code == 422


def test_detect_passes_score_threshold_to_engine() -> None:
    """score_threshold が engine.predict に伝搬する."""
    from unittest.mock import MagicMock

    engine = MagicMock()
    engine.backend_name = "pytorch"
    engine.predict.return_value = ([], {})
    set_engine(engine)
    app = create_app(None)
    payload = _encode_raw(np.zeros((4, 4, 3), dtype=np.uint8))
    payload["score_threshold"] = 0.7
    with TestClient(app) as client:
        res = client.post("/api/v1/detect", json=payload)
    assert res.status_code == 200
    engine.predict.assert_called_once()
    _, kwargs = engine.predict.call_args
    assert kwargs["score_threshold"] == 0.7


def test_detect_returns_gpu_metrics_when_pynvml_available() -> None:
    """pynvml 取得成功時, GPU clock / VRAM / 温度 がレスポンスに int で載る."""
    from types import SimpleNamespace

    _install_engine_returning([])
    fake_handle = object()
    fake_mem = SimpleNamespace(used=1024 * 1024 * 768)  # 768 MiB
    app = create_app(None)
    payload = _encode_raw(np.zeros((4, 4, 3), dtype=np.uint8))
    with (
        patch.object(pynvml, "nvmlInit", return_value=None),
        patch.object(pynvml, "nvmlDeviceGetHandleByIndex", return_value=fake_handle),
        patch.object(pynvml, "nvmlDeviceGetClockInfo", return_value=1800),
        patch.object(pynvml, "nvmlDeviceGetMemoryInfo", return_value=fake_mem),
        patch.object(pynvml, "nvmlDeviceGetTemperature", return_value=58),
        TestClient(app) as client,
    ):
        res = client.post("/api/v1/detect", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["gpu_clock_mhz"] == 1800
    assert body["gpu_vram_used_mb"] == 768
    assert body["gpu_temperature_c"] == 58


def test_detect_returns_null_gpu_metrics_when_pynvml_unavailable() -> None:
    """pynvml 初期化失敗時, 3 メトリクスともレスポンスで null になる."""
    _install_engine_returning([])
    app = create_app(None)
    payload = _encode_raw(np.zeros((4, 4, 3), dtype=np.uint8))
    with (
        patch.object(pynvml, "nvmlInit", side_effect=pynvml.NVMLError(1)),
        TestClient(app) as client,
    ):
        res = client.post("/api/v1/detect", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["gpu_clock_mhz"] is None
    assert body["gpu_vram_used_mb"] is None
    assert body["gpu_temperature_c"] is None
