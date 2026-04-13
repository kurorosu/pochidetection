"""create_app(None) 経由 (lifespan 無効) で 4 エンドポイントの応答を検証."""

from fastapi.testclient import TestClient

from pochidetection import __version__
from pochidetection.api import app as app_module
from pochidetection.api.app import create_app


def test_health_unhealthy_when_no_holder() -> None:
    """holder 未初期化時, /health は unhealthy + model_loaded=False を返す."""
    app_module._holder = None
    app = create_app(None)
    with TestClient(app) as client:
        res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "unhealthy"
    assert body["model_loaded"] is False


def test_version_includes_pochidetection_version() -> None:
    """/version は pochidetection.__version__ と api_version=v1 を返す."""
    app_module._holder = None
    app = create_app(None)
    with TestClient(app) as client:
        res = client.get("/api/v1/version")
    assert res.status_code == 200
    body = res.json()
    assert body["pochidetection_version"] == __version__
    assert body["api_version"] == "v1"
    assert "torch" in body["backend_versions"]


def test_backends_lists_pytorch_and_current_none() -> None:
    """holder 未初期化時, /backends.current=='none', available に pytorch 含む."""
    app_module._holder = None
    app = create_app(None)
    with TestClient(app) as client:
        res = client.get("/api/v1/backends")
    assert res.status_code == 200
    body = res.json()
    assert "pytorch" in body["available"]
    assert body["current"] == "none"


def test_model_info_returns_503_when_no_holder() -> None:
    """holder 未初期化時, /model-info は 503 を返す."""
    app_module._holder = None
    app = create_app(None)
    with TestClient(app) as client:
        res = client.get("/api/v1/model-info")
    assert res.status_code == 503
