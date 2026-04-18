"""検出推論エンドポイント `POST /api/v1/detect`.

`e2e_time_ms` は ``time.perf_counter()`` でリクエスト処理全体の wall clock を計測する.
レスポンスの ``phase_times_ms`` には pipeline 内訳 (preprocess / inference /
postprocess の ms 値) を返す. CUDA 利用時は ``pipeline_inference_gpu_ms``
(CUDA Event 計測) も追加され, wall-clock との差分が Python 側待ち時間の指標になる.
"""

import binascii
import time

from fastapi import APIRouter, HTTPException

from pochidetection.api.gpu_clock import get_gpu_clock_mhz
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


def _format_inference(phase_times: dict[str, float]) -> str:
    """Format the inference summary token for the INFO log line."""
    inf_wall = phase_times.get("pipeline_inference_ms")
    inf_gpu = phase_times.get("pipeline_inference_gpu_ms")
    if inf_gpu is not None and inf_wall is not None:
        return f"inf(gpu/wall)={inf_gpu:.1f}/{inf_wall:.1f}"
    if inf_wall is not None:
        return f"inf(wall)={inf_wall:.1f}"
    return "inf=N/A"


def _format_phase(phase_times: dict[str, float], key: str, label: str) -> str:
    value = phase_times.get(key)
    return f"{label}={value:.1f}" if value is not None else f"{label}=N/A"


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

    t_start = time.perf_counter()
    data = request.model_dump()

    try:
        serializer = _get_cached_serializer(request.format)
        image = serializer.deserialize(data)
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

    try:
        detections, predict_phases = engine.predict(
            image, score_threshold=request.score_threshold
        )
    except Exception as e:
        logger.exception("推論エラー")
        raise HTTPException(
            status_code=500,
            detail="推論中にエラーが発生しました",
        ) from e

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    phase_times: dict[str, float] = {k: round(v, 3) for k, v in predict_phases.items()}

    # Why: GPU clock は e2e 計測外 (logger.info 直前) で取得し,
    # pynvml 呼び出し (~数百 us) が e2e_time_ms に乗らないようにする.
    clk_mhz = get_gpu_clock_mhz()
    clk_str = f" clk={clk_mhz}MHz" if clk_mhz is not None else ""

    logger.info(
        f"Detection complete: detections={len(detections)} e2e={elapsed_ms:.1f} "
        f"{_format_phase(phase_times, 'pipeline_preprocess_ms', 'pre')} "
        f"{_format_inference(phase_times)} "
        f"{_format_phase(phase_times, 'pipeline_postprocess_ms', 'post')}"
        f"{clk_str} pipeline={engine.pipeline.pipeline_mode}"
    )

    return DetectResponse(
        detections=[DetectionDict(**d) for d in detections],
        e2e_time_ms=round(elapsed_ms, 3),
        backend=engine.backend_name,
        phase_times_ms=phase_times,
    )
