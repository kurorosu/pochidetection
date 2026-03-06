"""InferenceTimer のテスト."""

import time

import pytest

from pochidetection.utils import InferenceTimer


class TestInferenceTimer:
    """InferenceTimer クラスのテスト."""

    def test_cpu_timer_measures_time(self) -> None:
        """CPU タイマーが時間を計測することを確認."""
        timer = InferenceTimer(device="cpu")

        with timer.measure():
            time.sleep(0.01)  # 10ms

        # 10ms 以上であることを確認 (誤差を考慮), 上限 500ms
        assert 5.0 <= timer.last_time_ms < 500.0

    def test_count_increments(self) -> None:
        """計測回数がインクリメントされることを確認 (skip_first=False)."""
        timer = InferenceTimer(device="cpu", skip_first=False)

        assert timer.count == 0

        with timer.measure():
            pass
        assert timer.count == 1

        with timer.measure():
            pass
        assert timer.count == 2

    def test_skip_first_excludes_warmup(self) -> None:
        """skip_first=True で最初の計測がスキップされることを確認."""
        timer = InferenceTimer(device="cpu", skip_first=True)

        # 1回目 (スキップ)
        with timer.measure():
            pass
        assert timer.count == 0

        # 2回目 (カウント)
        with timer.measure():
            pass
        assert timer.count == 1

    def test_total_time_accumulates(self) -> None:
        """合計時間が累積されることを確認."""
        timer = InferenceTimer(device="cpu", skip_first=False)

        with timer.measure():
            time.sleep(0.01)

        with timer.measure():
            time.sleep(0.01)

        # 2回分の合計が 10ms 以上, 上限 1000ms
        assert 10.0 <= timer.total_time_ms < 1000.0

    def test_average_time_calculated(self) -> None:
        """平均時間が正しく計算されることを確認."""
        timer = InferenceTimer(device="cpu", skip_first=False)

        with timer.measure():
            time.sleep(0.01)

        with timer.measure():
            time.sleep(0.01)

        # 平均が total / count と一致
        expected = timer.total_time_ms / timer.count
        assert timer.average_time_ms == expected

    def test_average_time_zero_when_no_measurements(self) -> None:
        """計測なしの場合, 平均は 0 を返すことを確認."""
        timer = InferenceTimer(device="cpu")
        assert timer.average_time_ms == 0.0

    def test_measure_with_cuda_fallback(self) -> None:
        """CUDA モードでも measure() が動作することを確認."""
        timer = InferenceTimer(device="cuda", skip_first=False)

        with timer.measure():
            time.sleep(0.01)

        assert timer.count == 1
        assert 5.0 <= timer.last_time_ms < 500.0

    def test_cpu_fallback_when_cuda_unavailable(self) -> None:
        """CUDA が利用不可の場合, CPU にフォールバックすることを確認."""
        timer = InferenceTimer(device="cuda")

        # CUDA 有無に関わらず measure() が正常動作する
        with timer.measure():
            pass

        # フォールバック時でも計測が完了している
        assert timer.last_time_ms >= 0.0

    def test_reset_clears_accumulated_data(self) -> None:
        """reset() で累積データがクリアされることを確認."""
        timer = InferenceTimer(device="cpu", skip_first=False)

        # 計測実行
        with timer.measure():
            time.sleep(0.01)

        assert timer.count > 0
        assert timer.total_time_ms > 0

        # リセット
        timer.reset()

        assert timer.count == 0
        assert timer.total_time_ms == 0.0
        assert timer.last_time_ms == 0.0
        assert timer.average_time_ms == 0.0

    def test_context_manager_measures_time(self) -> None:
        """Context Manager で時間を計測できることを確認."""
        timer = InferenceTimer(device="cpu", skip_first=False)

        with timer.measure():
            time.sleep(0.01)

        assert timer.count == 1
        assert 5.0 <= timer.last_time_ms < 500.0

    def test_context_manager_stops_on_exception(self) -> None:
        """Context Manager が例外時にも stop() を呼ぶことを確認."""
        timer = InferenceTimer(device="cpu", skip_first=False)

        with pytest.raises(ValueError, match="test error"):
            with timer.measure():
                raise ValueError("test error")

        # 例外が発生しても stop() が呼ばれ, カウントされる
        assert timer.count == 1
