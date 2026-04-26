"""create_app(None) 経由 (lifespan 無効) で 4 エンドポイントの応答を検証."""

from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pochidetection import __version__
from pochidetection.api.app import BodySizeLimitMiddleware, create_app
from pochidetection.api.config import ServerConfig


def test_health_unhealthy_when_no_engine() -> None:
    """Engine 未初期化時, /health は unhealthy + model_loaded=False を返す."""
    app = create_app(None)
    with TestClient(app) as client:
        res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "unhealthy"
    assert body["model_loaded"] is False


def test_version_includes_pochidetection_version() -> None:
    """/version は pochidetection.__version__ と api_version=v1 を返す."""
    app = create_app(None)
    with TestClient(app) as client:
        res = client.get("/api/v1/version")
    assert res.status_code == 200
    body = res.json()
    assert body["pochidetection_version"] == __version__
    assert body["api_version"] == "v1"
    assert "torch" in body["backend_versions"]


def test_backends_lists_pytorch_and_current_none() -> None:
    """Engine 未初期化時, /backends.current=='none', available に pytorch 含む."""
    app = create_app(None)
    with TestClient(app) as client:
        res = client.get("/api/v1/backends")
    assert res.status_code == 200
    body = res.json()
    assert "pytorch" in body["available"]
    assert body["current"] == "none"


def test_model_info_returns_503_when_no_engine() -> None:
    """Engine 未初期化時, /model-info は 503 を返す."""
    app = create_app(None)
    with TestClient(app) as client:
        res = client.get("/api/v1/model-info")
    assert res.status_code == 503


def _build_echo_app(max_body_size: int) -> FastAPI:
    """body サイズ上限 middleware のみを載せた最小 echo app."""
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_body_size=max_body_size)  # type: ignore[arg-type]

    @app.post("/echo")
    async def echo(payload: dict) -> dict:  # type: ignore[type-arg]
        return {"size": len(payload.get("data", ""))}

    return app


def test_body_size_limit_allows_payload_within_limit() -> None:
    """上限以下の body は通常どおり処理される."""
    app = _build_echo_app(max_body_size=1024)
    with TestClient(app) as client:
        res = client.post("/echo", json={"data": "x" * 100})
    assert res.status_code == 200
    assert res.json() == {"size": 100}


def test_body_size_limit_rejects_oversized_payload() -> None:
    """Content-Length が上限超過の body は 413 で弾かれる."""
    app = _build_echo_app(max_body_size=128)
    with TestClient(app) as client:
        res = client.post("/echo", json={"data": "x" * 4096})
    assert res.status_code == 413
    body = res.json()
    assert "Request body too large" in body["detail"]
    assert "128" in body["detail"]


def test_body_size_limit_rejects_invalid_content_length() -> None:
    """Content-Length ヘッダーが非整数なら 413 で弾く."""
    app = _build_echo_app(max_body_size=1024)
    with TestClient(app) as client:
        res = client.post(
            "/echo",
            data=b"{}",  # type: ignore[arg-type]
            headers={
                "content-type": "application/json",
                "content-length": "not-a-number",
            },
        )
    assert res.status_code == 413


def test_body_size_limit_streaming_exceeds_limit() -> None:
    """Content-Length なし (chunked) でも累積バイト数で 413 判定する."""
    app = _build_echo_app(max_body_size=16)

    def gen() -> Iterator[bytes]:
        yield b'{"data":"'
        yield b"x" * 64
        yield b'"}'

    with TestClient(app) as client:
        res = client.post(
            "/echo",
            data=gen(),  # type: ignore[arg-type]
            headers={"content-type": "application/json"},
        )
    assert res.status_code == 413


def test_create_app_registers_body_size_middleware() -> None:
    """create_app() で BodySizeLimitMiddleware が登録されていることを確認."""
    app = create_app(None)
    names = {m.cls.__name__ for m in app.user_middleware}  # type: ignore[attr-defined]
    assert "BodySizeLimitMiddleware" in names


class TestServerConfigModelPath:
    """ServerConfig.model_path の None 許容 (pretrained 経路) を検証."""

    def test_accepts_none_for_pretrained(self) -> None:
        """model_path 省略時は None になる (pretrained フォールバック用)."""
        cfg = ServerConfig()
        assert cfg.model_path is None

    def test_accepts_explicit_path(self) -> None:
        """明示パス指定時はその Path 値を保持する."""
        cfg = ServerConfig(model_path=Path("work_dirs/dummy/best"))
        assert cfg.model_path == Path("work_dirs/dummy/best")
