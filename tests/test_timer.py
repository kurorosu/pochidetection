"""InferenceTimer のテスト."""

import time

import pytest

from pochidetection.utils import InferenceTimer


class TestInferenceTimer:
    """InferenceTimer クラスのテスト."""

    def test_cpu_timer_measures_time(self) -> None:
        """CPU タイマーが時間を計測することを確認."""
        timer = InferenceTimer(device="cpu")

        timer.start()
        time.sleep(0.01)  # 10ms
        elapsed = timer.stop()

        # 10ms 以上であることを確認 (誤差を考慮)
        assert elapsed >= 5.0

    def test_count_increments(self) -> None:
        """計測回数がインクリメントされることを確認 (skip_first=False)."""
        timer = InferenceTimer(device="cpu", skip_first=False)

        assert timer.count == 0

        timer.start()
        timer.stop()
        assert timer.count == 1

        timer.start()
        timer.stop()
        assert timer.count == 2

    def test_skip_first_excludes_warmup(self) -> None:
        """skip_first=True で最初の計測がスキップされることを確認."""
        timer = InferenceTimer(device="cpu", skip_first=True)

        # 1回目 (スキップ)
        timer.start()
        timer.stop()
        assert timer.count == 0

        # 2回目 (カウント)
        timer.start()
        timer.stop()
        assert timer.count == 1

    def test_total_time_accumulates(self) -> None:
        """合計時間が累積されることを確認."""
        timer = InferenceTimer(device="cpu", skip_first=False)

        timer.start()
        time.sleep(0.01)
        timer.stop()

        timer.start()
        time.sleep(0.01)
        timer.stop()

        # 2回分の合計が 10ms 以上
        assert timer.total_time_ms >= 10.0

    def test_average_time_calculated(self) -> None:
        """平均時間が正しく計算されることを確認."""
        timer = InferenceTimer(device="cpu", skip_first=False)

        timer.start()
        time.sleep(0.01)
        timer.stop()

        timer.start()
        time.sleep(0.01)
        timer.stop()

        # 平均が total / count と一致
        expected = timer.total_time_ms / timer.count
        assert timer.average_time_ms == expected

    def test_average_time_zero_when_no_measurements(self) -> None:
        """計測なしの場合, 平均は 0 を返すことを確認."""
        timer = InferenceTimer(device="cpu")
        assert timer.average_time_ms == 0.0

    def test_stop_without_start_raises_error_for_cuda(self) -> None:
        """CUDA モードで start() なしに stop() を呼ぶとエラー."""
        # CUDA が利用可能な場合のみテスト
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        timer = InferenceTimer(device="cuda")
        with pytest.raises(RuntimeError, match="start\\(\\) must be called"):
            timer.stop()

    def test_cpu_fallback_when_cuda_unavailable(self) -> None:
        """CUDA が利用不可の場合, CPU にフォールバックすることを確認."""
        import torch

        timer = InferenceTimer(device="cuda")

        # CUDA が利用不可なら _use_cuda は False
        if not torch.cuda.is_available():
            assert timer._use_cuda is False
        else:
            assert timer._use_cuda is True
