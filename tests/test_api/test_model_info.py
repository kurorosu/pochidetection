"""Engine 注入時の /model-info, /health, /backends の応答を検証."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from pochidetection.api.app import create_app
from pochidetection.api.state import set_engine


def _make_engine() -> MagicMock:
    engine = MagicMock()
    engine.backend_name = "pytorch"
    engine.get_model_info.return_value = {
        "architecture": "RTDetr",
        "num_classes": 2,
        "class_names": ["dog", "cat"],
        "input_size": (640, 640),
        "model_path": "/tmp/best",
        "backend": "pytorch",
    }
    return engine


def test_model_info_returns_engine_metadata() -> None:
    """Engine セット時, /model-info が architecture / num_classes / class_names を返す."""
    set_engine(_make_engine())
    app = create_app(None)
    with TestClient(app) as client:
        res = client.get("/api/v1/model-info")
    assert res.status_code == 200
    body = res.json()
    assert body["architecture"] == "RTDetr"
    assert body["num_classes"] == 2
    assert body["class_names"] == ["dog", "cat"]
    assert body["input_size"] == [640, 640]
    assert body["backend"] == "pytorch"


def test_health_returns_healthy_when_engine_set() -> None:
    """Engine セット時, /health が healthy + architecture を返す."""
    set_engine(_make_engine())
    app = create_app(None)
    with TestClient(app) as client:
        res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["architecture"] == "RTDetr"


def test_backends_current_reflects_engine() -> None:
    """Engine セット時, /backends.current が engine.backend_name を返す."""
    set_engine(_make_engine())
    app = create_app(None)
    with TestClient(app) as client:
        res = client.get("/api/v1/backends")
    assert res.status_code == 200
    assert res.json()["current"] == "pytorch"
