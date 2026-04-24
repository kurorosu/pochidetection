"""検出推論エンドポイント `POST /api/v1/detect`.

`e2e_time_ms` は ``time.perf_counter()`` でリクエスト処理全体の wall clock を計測する.
レスポンスの ``phase_times_ms`` には pipeline 内訳 (preprocess / inference /
postprocess の ms 値) を返す. CUDA 利用時は ``pipeline_inference_gpu_ms``
(CUDA Event 計測) も追加され, wall-clock との差分が Python 側待ち時間の指標になる.
"""

import binascii
import threading
import time

from fastapi import APIRouter, HTTPException

from pochidetection.api.gpu_metrics import (
    get_gpu_clock_mhz,
    get_gpu_temperature_c,
    get_gpu_vram_used_mb,
)
from pochidetection.api.log_format import format_inference, format_phase
from pochidetection.api.schemas import DetectionDict, DetectRequest, DetectResponse
from pochidetection.api.serializers import IImageSerializer, create_serializer
from pochidetection.api.state import get_engine
from pochidetection.logging import LoggerManager

logger = LoggerManager().get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["inference"])

# 複数 worker thread / 並列リクエストでの race condition を防ぐため lock で保護.
# キャッシュ hit 時のクリティカルセクションは dict lookup のみで十分軽量.
_serializer_cache: dict[str, IImageSerializer] = {}
_serializer_cache_lock = threading.Lock()


def _get_cached_serializer(fmt: str) -> IImageSerializer:
    """指定 format のシリアライザをキャッシュ経由で取得する (thread-safe).

    Args:
        fmt: 画像フォーマット名 (``"raw"`` / ``"jpeg"``).

    Returns:
        ``IImageSerializer`` 実装インスタンス.
    """
    with _serializer_cache_lock:
        serializer = _serializer_cache.get(fmt)
        if serializer is None:
            serializer = create_serializer(fmt)
            _serializer_cache[fmt] = serializer
        return serializer


@router.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest) -> DetectResponse:
    """1 枚の画像に対して検出を実行し, バウンディングボックスを返す.

    Args:
        request: base64 エンコードされた画像 / ``format`` / ``score_threshold`` を
            含む検出リクエスト.

    Returns:
        検出結果 (``class_id`` / ``class_name`` / ``confidence`` / ``bbox``), e2e
        タイミング (``e2e_time_ms``), フェーズ別タイミング (``phase_times_ms``),
        backend 名を含むレスポンス.

    Raises:
        HTTPException: 503 (モデル未ロード) / 400 (画像デシリアライズ失敗) /
            500 (推論または予期しないエラー) のいずれか.
    """
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

    # Why: GPU メトリクスは e2e 計測外 (logger.info 直前) で取得し,
    # pynvml 呼び出し (~数百 us x 3) が e2e_time_ms に乗らないようにする.
    clk_mhz = get_gpu_clock_mhz()
    vram_mb = get_gpu_vram_used_mb()
    temp_c = get_gpu_temperature_c()
    clk_str = f" clk={clk_mhz}MHz" if clk_mhz is not None else ""

    logger.info(
        f"Detection complete: detections={len(detections)} e2e={elapsed_ms:.1f} "
        f"{format_phase(phase_times, 'pipeline_preprocess_ms', 'pre')} "
        f"{format_inference(phase_times)} "
        f"{format_phase(phase_times, 'pipeline_postprocess_ms', 'post')}"
        f"{clk_str} pipeline={engine.pipeline.pipeline_mode}"
    )

    return DetectResponse(
        detections=[DetectionDict(**d) for d in detections],
        e2e_time_ms=round(elapsed_ms, 3),
        backend=engine.backend_name,
        phase_times_ms=phase_times,
        gpu_clock_mhz=clk_mhz,
        gpu_vram_used_mb=vram_mb,
        gpu_temperature_c=temp_c,
    )
