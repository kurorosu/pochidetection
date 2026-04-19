"""health, version, model-info, backends エンドポイント."""

from fastapi import APIRouter, HTTPException

from pochidetection import __version__
from pochidetection.api.backends import _safe_version, get_available_backends
from pochidetection.api.schemas import (
    BackendsResponse,
    HealthResponse,
    ModelInfoResponse,
    VersionResponse,
)
from pochidetection.api.state import get_engine

router = APIRouter(prefix="/api/v1")

API_VERSION = "v1"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return server / model status."""
    try:
        engine = get_engine()
    except RuntimeError:
        return HealthResponse(status="unhealthy", model_loaded=False)
    info = engine.get_model_info()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        architecture=info["architecture"],
    )


@router.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    """Return version info for pochidetection and the runtime backends."""
    backend_versions: dict[str, str] = {}
    for pkg in ("torch", "onnxruntime-gpu", "onnxruntime", "tensorrt"):
        v = _safe_version(pkg)
        if v is not None:
            backend_versions[pkg] = v

    return VersionResponse(
        pochidetection_version=__version__,
        api_version=API_VERSION,
        backend_versions=backend_versions,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return loaded-model metadata.

    Raises:
        HTTPException: モデル未ロード時に 503 を返す.
    """
    try:
        engine = get_engine()
    except RuntimeError as e:
        raise HTTPException(
            status_code=503, detail="モデルが初期化されていません"
        ) from e
    info = engine.get_model_info()
    return ModelInfoResponse(
        architecture=info["architecture"],
        num_classes=info["num_classes"],
        class_names=info["class_names"],
        input_size=info["input_size"],
        model_path=info["model_path"],
        backend=info["backend"],
    )


@router.get("/backends", response_model=BackendsResponse)
def backends() -> BackendsResponse:
    """Return available backends and the currently loaded one."""
    try:
        current = get_engine().backend_name
    except RuntimeError:
        current = "none"
    return BackendsResponse(
        available=get_available_backends(),
        current=current,
    )
