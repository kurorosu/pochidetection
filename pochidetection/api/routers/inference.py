"""検出推論エンドポイント `POST /api/v1/detect`.

`e2e_time_ms` は ``time.perf_counter()`` で ``engine.predict()`` 呼び出しの wall clock
(デコード後の推論 + 後処理 + フィルタ) を計測する.
"""

import binascii
import time

from fastapi import APIRouter, HTTPException

from pochidetection.api.schemas import DetectionDict, DetectRequest, DetectResponse
from pochidetection.api.serializers import IImageSerializer, create_serializer
from pochidetection.logging import LoggerManager

logger = LoggerManager().get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["inference"])

_serializer_cache: dict[str, IImageSerializer] = {}


def _get_cached_serializer(fmt: str) -> IImageSerializer:
    """Return a cached serializer for the given format."""
    serializer = _serializer_cache.get(fmt)
    if serializer is None:
        serializer = create_serializer(fmt)
        _serializer_cache[fmt] = serializer
    return serializer


@router.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest) -> DetectResponse:
    """Run detection on a single image and return bounding boxes."""
    from pochidetection.api.app import get_engine

    try:
        engine = get_engine()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="モデルがロードされていません",
        )

    try:
        serializer = _get_cached_serializer(request.format)
        image = serializer.deserialize(request.model_dump())
    except (ValueError, binascii.Error) as e:
        # Why: base64.b64decode は不正入力で binascii.Error を投げる. ValueError と同様に
        # クライアント起因のエラーとして 400 で返す.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("画像デシリアライズエラー")
        raise HTTPException(
            status_code=500,
            detail="画像処理中にエラーが発生しました",
        ) from e

    start = time.perf_counter()
    try:
        detections = engine.predict(image, score_threshold=request.score_threshold)
    except Exception as e:
        logger.exception("推論エラー")
        raise HTTPException(
            status_code=500,
            detail="推論中にエラーが発生しました",
        ) from e
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        f"Detection complete: {len(detections)} objects, e2e={elapsed_ms:.1f}ms"
    )

    return DetectResponse(
        detections=[DetectionDict(**d) for d in detections],
        e2e_time_ms=round(elapsed_ms, 3),
        backend=engine.backend_name,
    )
