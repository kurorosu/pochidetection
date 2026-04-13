"""health, version, model-info, backends エンドポイント."""

import importlib.metadata

from fastapi import APIRouter, HTTPException

from pochidetection import __version__
from pochidetection.api import app as app_module
from pochidetection.api.schemas import (
    BackendsResponse,
    HealthResponse,
    ModelInfoResponse,
    VersionResponse,
)

router = APIRouter(prefix="/api/v1")

API_VERSION = "v1"


def _safe_version(package: str) -> str | None:
    """インストール済みパッケージのバージョンを取得. 未インストールなら None."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _detect_available_backends() -> list[str]:
    """利用可能なバックエンドを動的に検出する."""
    available: list[str] = []
    if _safe_version("torch") is not None:
        available.append("pytorch")
    if (
        _safe_version("onnxruntime-gpu") is not None
        or _safe_version("onnxruntime") is not None
    ):
        available.append("onnx")
    if _safe_version("tensorrt") is not None:
        available.append("tensorrt")
    return available


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """ヘルスチェック."""
    if app_module._holder is None:
        return HealthResponse(status="unhealthy", model_loaded=False)
    holder = app_module._holder
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        architecture=holder.architecture,
    )


@router.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    """バージョン情報."""
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
    """モデル情報を返却する.

    Raises:
        HTTPException: モデル未ロード時に 503 を返す.
    """
    if app_module._holder is None:
        raise HTTPException(status_code=503, detail="モデルが初期化されていません")
    holder = app_module._holder
    return ModelInfoResponse(
        architecture=holder.architecture,
        num_classes=holder.num_classes,
        class_names=holder.class_names,
        input_size=holder.input_size,
        model_path=holder.model_path,
        backend=holder.backend_name,
    )


@router.get("/backends", response_model=BackendsResponse)
def backends() -> BackendsResponse:
    """利用可能なバックエンド一覧と現在のバックエンドを返却する."""
    current = app_module._holder.backend_name if app_module._holder else "none"
    return BackendsResponse(
        available=_detect_available_backends(),
        current=current,
    )
