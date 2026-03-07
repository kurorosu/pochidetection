"""ベンチマーク結果スキーマ・ビルダー・エクスポーターのテスト."""

import json
import re
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from pochidetection.scripts.common.inference import _log_benchmark_summary
from pochidetection.utils import PhasedTimer
from pochidetection.utils.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkSamples,
    DetectionMetrics,
    PhaseMetrics,
    build_benchmark_result,
    write_benchmark_result,
)

PHASES = ["preprocess", "inference", "postprocess"]


# ---------- スキーマテスト ----------


class TestPhaseMetrics:
    """PhaseMetrics のフィールド検証テスト."""

    def test_valid_construction(self) -> None:
        """正常な値で構築できることを確認."""
        m = PhaseMetrics(total_ms=100.0, count=10, average_ms=10.0)
        assert m.total_ms == 100.0
        assert m.count == 10
        assert m.average_ms == 10.0

    def test_extra_field_raises_validation_error(self) -> None:
        """extra="forbid" で未知フィールドが拒否されることを確認."""
        with pytest.raises(ValidationError):
            PhaseMetrics(total_ms=1.0, count=1, average_ms=1.0, unknown=1.0)  # type: ignore[call-arg]


class TestBenchmarkMetrics:
    """BenchmarkMetrics のフィールド検証テスト."""

    def test_valid_construction_with_phases(self) -> None:
        """phases dict を含む正常な構築を確認."""
        phases = {
            "inference": PhaseMetrics(total_ms=50.0, count=5, average_ms=10.0),
        }
        m = BenchmarkMetrics(
            avg_inference_ms=10.0,
            avg_e2e_ms=20.0,
            throughput_inference_ips=100.0,
            throughput_e2e_ips=50.0,
            phases=phases,
        )
        assert m.phases["inference"].count == 5

    def test_extra_field_raises_validation_error(self) -> None:
        """extra="forbid" で未知フィールドが拒否されることを確認."""
        with pytest.raises(ValidationError):
            BenchmarkMetrics(
                avg_inference_ms=1.0,
                avg_e2e_ms=1.0,
                throughput_inference_ips=1.0,
                throughput_e2e_ips=1.0,
                phases={},
                unknown=1.0,  # type: ignore[call-arg]
            )


class TestBenchmarkSamples:
    """BenchmarkSamples のフィールド検証テスト."""

    def test_valid_construction(self) -> None:
        """正常な値で構築できることを確認."""
        s = BenchmarkSamples(num_samples=10, measured_samples=9, warmup_samples=1)
        assert s.num_samples == 10


class TestBenchmarkResult:
    """BenchmarkResult のラウンドトリップテスト."""

    def test_model_dump_json_round_trip(self) -> None:
        """model_dump_json() → JSON パース → model_validate で復元できることを確認."""
        phases = {
            "preprocess": PhaseMetrics(total_ms=10.0, count=5, average_ms=2.0),
            "inference": PhaseMetrics(total_ms=50.0, count=5, average_ms=10.0),
            "postprocess": PhaseMetrics(total_ms=5.0, count=5, average_ms=1.0),
        }
        original = BenchmarkResult(
            timestamp_jst="2026-03-01 12:00:00",
            device="cpu",
            precision="fp32",
            model_path="/tmp/model",
            metrics=BenchmarkMetrics(
                avg_inference_ms=10.0,
                avg_e2e_ms=13.0,
                throughput_inference_ips=100.0,
                throughput_e2e_ips=76.9,
                phases=phases,
            ),
            samples=BenchmarkSamples(
                num_samples=6, measured_samples=5, warmup_samples=1
            ),
        )

        json_str = original.model_dump_json(indent=2)
        parsed = json.loads(json_str)
        restored = BenchmarkResult.model_validate(parsed)

        assert restored == original
        assert restored.schema_version == BENCHMARK_SCHEMA_VERSION

    def test_extra_field_raises_validation_error(self) -> None:
        """extra="forbid" で未知フィールドが拒否されることを確認."""
        with pytest.raises(ValidationError):
            BenchmarkResult(
                timestamp_jst="2026-03-01 12:00:00",
                device="cpu",
                precision="fp32",
                model_path="/tmp/model",
                metrics=BenchmarkMetrics(
                    avg_inference_ms=1.0,
                    avg_e2e_ms=1.0,
                    throughput_inference_ips=1.0,
                    throughput_e2e_ips=1.0,
                    phases={},
                ),
                samples=BenchmarkSamples(
                    num_samples=1, measured_samples=1, warmup_samples=0
                ),
                unknown="bad",  # type: ignore[call-arg]
            )


class TestDetectionMetrics:
    """DetectionMetrics のテスト."""

    def test_valid_construction(self) -> None:
        """正常な値で構築できることを確認."""
        m = DetectionMetrics(map_50=0.85, map_50_95=0.65)
        assert m.map_50 == 0.85
        assert m.map_50_95 == 0.65

    def test_extra_field_raises_validation_error(self) -> None:
        """extra="forbid" で未知フィールドが拒否されることを確認."""
        with pytest.raises(ValidationError):
            DetectionMetrics(map_50=0.5, map_50_95=0.3, unknown=1.0)  # type: ignore[call-arg]


class TestBenchmarkResultWithDetectionMetrics:
    """DetectionMetrics 付き BenchmarkResult のテスト."""

    def test_round_trip_with_detection_metrics(self) -> None:
        """detection_metrics 付き BenchmarkResult が JSON ラウンドトリップできることを確認."""
        phases = {
            "inference": PhaseMetrics(total_ms=50.0, count=5, average_ms=10.0),
        }
        original = BenchmarkResult(
            timestamp_jst="2026-03-01 12:00:00",
            device="cpu",
            precision="fp32",
            model_path="/tmp/model",
            metrics=BenchmarkMetrics(
                avg_inference_ms=10.0,
                avg_e2e_ms=10.0,
                throughput_inference_ips=100.0,
                throughput_e2e_ips=100.0,
                phases=phases,
            ),
            samples=BenchmarkSamples(
                num_samples=5, measured_samples=5, warmup_samples=0
            ),
            detection_metrics=DetectionMetrics(map_50=0.85, map_50_95=0.65),
        )

        json_str = original.model_dump_json(indent=2)
        parsed = json.loads(json_str)
        restored = BenchmarkResult.model_validate(parsed)

        assert restored == original
        assert restored.detection_metrics is not None
        assert restored.detection_metrics.map_50 == 0.85
        assert restored.detection_metrics.map_50_95 == 0.65

    def test_round_trip_without_detection_metrics(self) -> None:
        """detection_metrics が None の BenchmarkResult が JSON ラウンドトリップできることを確認."""
        phases = {
            "inference": PhaseMetrics(total_ms=50.0, count=5, average_ms=10.0),
        }
        original = BenchmarkResult(
            timestamp_jst="2026-03-01 12:00:00",
            device="cpu",
            precision="fp32",
            model_path="/tmp/model",
            metrics=BenchmarkMetrics(
                avg_inference_ms=10.0,
                avg_e2e_ms=10.0,
                throughput_inference_ips=100.0,
                throughput_e2e_ips=100.0,
                phases=phases,
            ),
            samples=BenchmarkSamples(
                num_samples=5, measured_samples=5, warmup_samples=0
            ),
        )

        json_str = original.model_dump_json(indent=2)
        parsed = json.loads(json_str)
        restored = BenchmarkResult.model_validate(parsed)

        assert restored == original
        assert restored.detection_metrics is None


# ---------- ビルダーテスト ----------


class TestBuildBenchmarkResult:
    """build_benchmark_result のテスト."""

    def test_builds_from_phased_timer(self) -> None:
        """PhasedTimer の計測結果から正しく構築されることを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu", skip_first=False)

        for _ in range(3):
            with timer.measure("preprocess"):
                time.sleep(0.001)
            with timer.measure("inference"):
                time.sleep(0.002)
            with timer.measure("postprocess"):
                time.sleep(0.001)

        result = build_benchmark_result(
            phased_timer=timer,
            num_images=3,
            device="cpu",
            precision="fp32",
            model_path="/tmp/model",
        )

        assert result.device == "cpu"
        assert result.precision == "fp32"
        assert result.model_path == "/tmp/model"
        assert result.schema_version == BENCHMARK_SCHEMA_VERSION

        # timestamp_jst の形式を検証
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result.timestamp_jst)

        # フェーズ
        assert set(result.metrics.phases.keys()) == set(PHASES)
        for phase in PHASES:
            assert result.metrics.phases[phase].count == 3

        # サンプル数
        assert result.samples.num_samples == 3
        assert result.samples.measured_samples == 3
        assert result.samples.warmup_samples == 0

        # スループット > 0
        assert result.metrics.avg_inference_ms > 0.0
        assert result.metrics.avg_e2e_ms > 0.0
        assert result.metrics.throughput_inference_ips > 0.0
        assert result.metrics.throughput_e2e_ips > 0.0

    def test_avg_e2e_is_total_divided_by_count(self) -> None:
        """avg_e2e_ms が全フェーズの total_ms 合計 / count であることを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu", skip_first=False)

        for _ in range(2):
            for phase in PHASES:
                with timer.measure(phase):
                    time.sleep(0.001)

        result = build_benchmark_result(
            phased_timer=timer,
            num_images=2,
            device="cpu",
            precision="fp32",
            model_path="/tmp/model",
        )

        total_e2e = sum(result.metrics.phases[p].total_ms for p in PHASES)
        count = result.metrics.phases[PHASES[0]].count
        expected_e2e = total_e2e / count
        assert result.metrics.avg_e2e_ms == pytest.approx(expected_e2e)


class TestBuildBenchmarkResultZeroDivision:
    """ゼロ除算ケースのテスト."""

    def test_zero_measured_returns_zero_throughput(self) -> None:
        """num_samples==1, skip_first==True で throughput が 0.0 になることを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu", skip_first=True)

        # 1回だけ計測 → skip_first により count==0
        for phase in PHASES:
            with timer.measure(phase):
                pass

        result = build_benchmark_result(
            phased_timer=timer,
            num_images=1,
            device="cpu",
            precision="fp32",
            model_path="/tmp/model",
        )

        assert result.metrics.avg_inference_ms == 0.0
        assert result.metrics.avg_e2e_ms == 0.0
        assert result.metrics.throughput_inference_ips == 0.0
        assert result.metrics.throughput_e2e_ips == 0.0
        assert result.samples.measured_samples == 0
        assert result.samples.warmup_samples == 1


class TestBuildBenchmarkResultWithWarmup:
    """warmup 込みのテスト."""

    def test_warmup_skipped_in_count(self) -> None:
        """skip_first=True で最初の計測がスキップされることを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu", skip_first=True)

        for _ in range(3):
            for phase in PHASES:
                with timer.measure(phase):
                    time.sleep(0.001)

        result = build_benchmark_result(
            phased_timer=timer,
            num_images=3,
            device="cpu",
            precision="fp32",
            model_path="/tmp/model",
        )

        assert result.samples.num_samples == 3
        assert result.samples.measured_samples == 2
        assert result.samples.warmup_samples == 1

        for phase in PHASES:
            assert result.metrics.phases[phase].count == 2


# ---------- エクスポーターテスト ----------


class TestWriteBenchmarkResult:
    """write_benchmark_result のテスト."""

    def test_writes_json_and_round_trips(self, tmp_path: Path) -> None:
        """JSON ファイルを書き出し, 読み戻しで復元できることを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu", skip_first=False)

        with timer.measure("preprocess"):
            pass
        with timer.measure("inference"):
            pass
        with timer.measure("postprocess"):
            pass

        result = build_benchmark_result(
            phased_timer=timer,
            num_images=1,
            device="cpu",
            precision="fp32",
            model_path="/tmp/model",
        )

        output_path = write_benchmark_result(tmp_path, result)

        assert output_path.exists()
        assert output_path.name == "benchmark_result.json"

        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        restored = BenchmarkResult.model_validate(loaded)
        assert restored == result


# ---------- _log_benchmark_summary テスト ----------


class TestLogBenchmarkSummary:
    """_log_benchmark_summary のテスト."""

    def test_runs_without_error(self) -> None:
        """正常な BenchmarkResult でエラーなく実行されることを確認."""
        phases = {
            "preprocess": PhaseMetrics(total_ms=10.0, count=5, average_ms=2.0),
            "inference": PhaseMetrics(total_ms=50.0, count=5, average_ms=10.0),
            "postprocess": PhaseMetrics(total_ms=5.0, count=5, average_ms=1.0),
        }
        result = BenchmarkResult(
            timestamp_jst="2026-03-01 12:00:00",
            device="cpu",
            precision="fp32",
            model_path="/tmp/model",
            metrics=BenchmarkMetrics(
                avg_inference_ms=10.0,
                avg_e2e_ms=13.0,
                throughput_inference_ips=100.0,
                throughput_e2e_ips=76.9,
                phases=phases,
            ),
            samples=BenchmarkSamples(
                num_samples=6, measured_samples=5, warmup_samples=1
            ),
        )

        # 例外が発生しないことを確認
        _log_benchmark_summary(result)


class TestLogBenchmarkSummaryZeroMeasured:
    """measured_samples==0 での _log_benchmark_summary テスト."""

    def test_zero_measured_runs_without_error(self) -> None:
        """全値 0.0 の BenchmarkResult でもエラーなく実行されることを確認."""
        phases = {
            "preprocess": PhaseMetrics(total_ms=0.0, count=0, average_ms=0.0),
            "inference": PhaseMetrics(total_ms=0.0, count=0, average_ms=0.0),
            "postprocess": PhaseMetrics(total_ms=0.0, count=0, average_ms=0.0),
        }
        result = BenchmarkResult(
            timestamp_jst="2026-03-01 12:00:00",
            device="cpu",
            precision="fp32",
            model_path="/tmp/model",
            metrics=BenchmarkMetrics(
                avg_inference_ms=0.0,
                avg_e2e_ms=0.0,
                throughput_inference_ips=0.0,
                throughput_e2e_ips=0.0,
                phases=phases,
            ),
            samples=BenchmarkSamples(
                num_samples=1, measured_samples=0, warmup_samples=1
            ),
        )

        # 例外が発生しないことを確認
        _log_benchmark_summary(result)
