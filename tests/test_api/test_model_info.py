"""ModelHolder 注入時の /model-info, /health の応答を検証."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from pochidetection.api import app as app_module
from pochidetection.api.app import ModelHolder, create_app


def _make_holder() -> ModelHolder:
    return ModelHolder(
        pipeline=MagicMock(),
        config={},  # type: ignore[typeddict-item]
        architecture="RTDetr",
        class_names=["dog", "cat"],
        num_classes=2,
        input_size=(640, 640),
        model_path="/tmp/best",
        backend_name="pytorch",
    )


def test_model_info_returns_holder_metadata() -> None:
    """holder セット時, /model-info が architecture / num_classes / class_names を返す."""
    app_module._holder = _make_holder()
    try:
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
    finally:
        app_module._holder = None


def test_health_returns_healthy_when_holder_set() -> None:
    """holder セット時, /health が healthy + architecture を返す."""
    app_module._holder = _make_holder()
    try:
        app = create_app(None)
        with TestClient(app) as client:
            res = client.get("/api/v1/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["architecture"] == "RTDetr"
    finally:
        app_module._holder = None


def test_backends_current_reflects_holder() -> None:
    """holder セット時, /backends.current が holder.backend_name を返す."""
    app_module._holder = _make_holder()
    try:
        app = create_app(None)
        with TestClient(app) as client:
            res = client.get("/api/v1/backends")
        assert res.status_code == 200
        assert res.json()["current"] == "pytorch"
    finally:
        app_module._holder = None
