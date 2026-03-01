"""PhasedTimer のテスト."""

import time

import pytest

from pochidetection.utils import PhasedTimer

PHASES = ["preprocess", "inference", "postprocess"]


class TestPhasedTimerInit:
    """PhasedTimer の初期化テスト."""

    def test_creates_timers_for_each_phase(self) -> None:
        """各フェーズに対応するタイマーが生成されることを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu")
        assert timer.phases == PHASES

    def test_empty_phases_raises_value_error(self) -> None:
        """空のフェーズリストで ValueError が発生することを確認."""
        with pytest.raises(ValueError, match="phases must not be empty"):
            PhasedTimer(phases=[], device="cpu")

    def test_duplicate_phases_raises_value_error(self) -> None:
        """重複フェーズで ValueError が発生することを確認."""
        with pytest.raises(ValueError, match="phases must not contain duplicates"):
            PhasedTimer(phases=["inference", "inference"], device="cpu")


class TestPhasedTimerMeasure:
    """PhasedTimer のフェーズ別計測テスト."""

    def test_measure_records_time_per_phase(self) -> None:
        """各フェーズが独立して時間を計測することを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu", skip_first=False)

        with timer.measure("preprocess"):
            time.sleep(0.01)

        with timer.measure("inference"):
            time.sleep(0.01)

        preprocess = timer.get_timer("preprocess")
        inference = timer.get_timer("inference")
        postprocess = timer.get_timer("postprocess")

        assert preprocess.count == 1
        assert preprocess.last_time_ms >= 5.0
        assert inference.count == 1
        assert inference.last_time_ms >= 5.0
        assert postprocess.count == 0

    def test_measure_unknown_phase_raises_value_error(self) -> None:
        """未登録フェーズで ValueError が発生することを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu")

        with pytest.raises(ValueError, match="Unknown phase: 'unknown'"):
            with timer.measure("unknown"):
                pass

    def test_measure_stops_on_exception(self) -> None:
        """例外発生時にも計測が完了することを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu", skip_first=False)

        with pytest.raises(ValueError, match="test error"):
            with timer.measure("inference"):
                raise ValueError("test error")

        assert timer.get_timer("inference").count == 1


class TestPhasedTimerGetTimer:
    """PhasedTimer の get_timer テスト."""

    def test_get_timer_returns_inference_timer(self) -> None:
        """get_timer が InferenceTimer を返すことを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu")
        t = timer.get_timer("preprocess")

        # InferenceTimer のプロパティにアクセスできる
        assert t.count == 0
        assert t.total_time_ms == 0.0

    def test_get_timer_unknown_phase_raises_value_error(self) -> None:
        """未登録フェーズで ValueError が発生することを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu")

        with pytest.raises(ValueError, match="Unknown phase: 'unknown'"):
            timer.get_timer("unknown")


class TestPhasedTimerSummary:
    """PhasedTimer の summary テスト."""

    def test_summary_returns_dict_with_all_phases(self) -> None:
        """summary が全フェーズを含む辞書を返すことを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu", skip_first=False)

        with timer.measure("preprocess"):
            time.sleep(0.01)

        with timer.measure("inference"):
            time.sleep(0.01)

        result = timer.summary()

        assert set(result.keys()) == set(PHASES)
        for phase in PHASES:
            assert "total_ms" in result[phase]
            assert "count" in result[phase]
            assert "average_ms" in result[phase]

    def test_summary_count_is_int(self) -> None:
        """summary の count が int 型であることを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu", skip_first=False)

        with timer.measure("inference"):
            pass

        result = timer.summary()
        assert isinstance(result["inference"]["count"], int)

    def test_summary_values_reflect_measurements(self) -> None:
        """summary の値が実際の計測結果を反映していることを確認."""
        timer = PhasedTimer(phases=["a", "b"], device="cpu", skip_first=False)

        with timer.measure("a"):
            time.sleep(0.01)

        with timer.measure("a"):
            time.sleep(0.01)

        result = timer.summary()
        assert result["a"]["count"] == 2
        assert result["a"]["total_ms"] >= 10.0
        assert result["b"]["count"] == 0
        assert result["b"]["average_ms"] == 0.0


class TestPhasedTimerReset:
    """PhasedTimer の reset テスト."""

    def test_reset_clears_all_phases(self) -> None:
        """reset で全フェーズの累積データがクリアされることを確認."""
        timer = PhasedTimer(phases=PHASES, device="cpu", skip_first=False)

        for phase in PHASES:
            with timer.measure(phase):
                time.sleep(0.005)

        timer.reset()

        for phase in PHASES:
            t = timer.get_timer(phase)
            assert t.count == 0
            assert t.total_time_ms == 0.0
            assert t.average_time_ms == 0.0


class TestPhasedTimerSkipFirst:
    """PhasedTimer の skip_first テスト."""

    def test_skip_first_applies_to_each_phase_independently(self) -> None:
        """skip_first が各フェーズに独立して適用されることを確認."""
        timer = PhasedTimer(phases=["a", "b"], device="cpu", skip_first=True)

        # a: 1回目 (スキップ) + 2回目 (カウント)
        with timer.measure("a"):
            pass
        with timer.measure("a"):
            pass

        # b: 1回目 (スキップ) のみ
        with timer.measure("b"):
            pass

        assert timer.get_timer("a").count == 1
        assert timer.get_timer("b").count == 0
