"""ベンチマーク結果スキーマ・構築・出力."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from pochidetection.utils.phased_timer import PhasedTimer

BENCHMARK_RESULT_FILENAME = "benchmark_result.json"
BENCHMARK_SCHEMA_VERSION = "1.0.0"

JST = timezone(timedelta(hours=9))


class PhaseMetrics(BaseModel):
    """フェーズ別計測メトリクス."""

    model_config = ConfigDict(extra="forbid")

    total_ms: float
    count: int
    average_ms: float


class BenchmarkMetrics(BaseModel):
    """ベンチマーク計測メトリクス."""

    model_config = ConfigDict(extra="forbid")

    avg_inference_ms: float
    avg_e2e_ms: float
    throughput_inference_ips: float
    throughput_e2e_ips: float
    phases: dict[str, PhaseMetrics]


class BenchmarkSamples(BaseModel):
    """サンプル数関連."""

    model_config = ConfigDict(extra="forbid")

    num_samples: int
    measured_samples: int
    warmup_samples: int


class DetectionMetrics(BaseModel):
    """精度評価メトリクス (mAP)."""

    model_config = ConfigDict(extra="forbid")

    map_50: float
    map_50_95: float


class BenchmarkResult(BaseModel):
    """benchmark_result.json のルートスキーマ."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = BENCHMARK_SCHEMA_VERSION
    timestamp_jst: str
    device: str
    precision: str
    model_path: str
    metrics: BenchmarkMetrics
    samples: BenchmarkSamples
    detection_metrics: DetectionMetrics | None = None


def _safe_throughput(avg_ms: float) -> float:
    """スループットを計算する. avg_ms <= 0.0 のとき 0.0 を返す.

    Args:
        avg_ms: 平均時間 (ミリ秒).

    Returns:
        スループット (images per second).
    """
    return 1000.0 / avg_ms if avg_ms > 0.0 else 0.0


def build_benchmark_result(
    phased_timer: PhasedTimer,
    num_images: int,
    device: str,
    precision: str,
    model_path: str,
    detection_metrics: DetectionMetrics | None = None,
) -> BenchmarkResult:
    """Phasedtimer の計測結果から BenchmarkResult を構築.

    スループット計算規則:
      throughput = 1000.0 / avg_ms if avg_ms > 0.0 else 0.0
    avg_inference_ms, avg_e2e_ms の両方に同じルールを適用する.
    InferenceTimer.average_time_ms は count==0 で 0.0 を返すため,
    num_samples==1 かつ skip_first==True の場合に発生し得る.

    Args:
        phased_timer: 計測済みの PhasedTimer インスタンス.
        num_images: 推論した画像総数.
        device: 実行デバイス.
        precision: 精度 ("fp32" or "fp16").
        model_path: モデルディレクトリのパス文字列.
        detection_metrics: 精度評価メトリクス. None の場合は含めない.

    Returns:
        構築した BenchmarkResult.
    """
    summary = phased_timer.summary()

    phases = {
        phase: PhaseMetrics(
            total_ms=data["total_ms"],
            count=int(data["count"]),
            average_ms=data["average_ms"],
        )
        for phase, data in summary.items()
    }

    avg_inference_ms = phases["inference"].average_ms if "inference" in phases else 0.0

    # E2E 平均: 全フェーズの total_ms 合計を計測サンプル数で割る.
    # 各フェーズの average_ms を足す方式だと, フェーズ間で count が
    # 異なる場合に正しい per-image E2E 時間にならないため.
    measured = next(iter(phases.values())).count if phases else 0
    total_e2e_ms = sum(p.total_ms for p in phases.values())
    avg_e2e_ms = total_e2e_ms / measured if measured > 0 else 0.0
    warmup = num_images - measured

    return BenchmarkResult(
        timestamp_jst=datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
        device=device,
        precision=precision,
        model_path=model_path,
        metrics=BenchmarkMetrics(
            avg_inference_ms=avg_inference_ms,
            avg_e2e_ms=avg_e2e_ms,
            throughput_inference_ips=_safe_throughput(avg_inference_ms),
            throughput_e2e_ips=_safe_throughput(avg_e2e_ms),
            phases=phases,
        ),
        samples=BenchmarkSamples(
            num_samples=num_images,
            measured_samples=measured,
            warmup_samples=warmup,
        ),
        detection_metrics=detection_metrics,
    )


def write_benchmark_result(
    output_dir: Path,
    result: BenchmarkResult,
    filename: str = BENCHMARK_RESULT_FILENAME,
) -> Path:
    """Benchmarkresult を JSON ファイルに書き出す.

    Args:
        output_dir: 出力ディレクトリ.
        result: ベンチマーク結果.
        filename: 出力ファイル名.

    Returns:
        書き出したファイルのパス.
    """
    output_path = output_dir / filename
    output_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return output_path
