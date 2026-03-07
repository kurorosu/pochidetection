"""EarlyStopping のテスト."""

import pytest

from pochidetection.utils import EarlyStopping


class TestEarlyStopping:
    """EarlyStopping クラスのテスト."""

    def test_max_mode_no_improvement_triggers_stop(self) -> None:
        """max モードで patience 回連続改善なしなら停止."""
        es = EarlyStopping(patience=3, mode="max")

        assert not es.step(0.5, epoch=1)  # best=0.5
        assert not es.step(0.4, epoch=2)  # counter=1
        assert not es.step(0.3, epoch=3)  # counter=2
        assert es.step(0.2, epoch=4)  # counter=3 → 停止

    def test_max_mode_improvement_resets_counter(self) -> None:
        """max モードで改善があればカウンタリセット."""
        es = EarlyStopping(patience=2, mode="max")

        assert not es.step(0.5, epoch=1)
        assert not es.step(0.4, epoch=2)  # counter=1
        assert not es.step(0.6, epoch=3)  # 改善 → counter=0
        assert es.counter == 0
        assert es.best_value == 0.6
        assert es.best_epoch == 3

    def test_min_mode_no_improvement_triggers_stop(self) -> None:
        """min モードで patience 回連続改善なしなら停止."""
        es = EarlyStopping(patience=2, mode="min")

        assert not es.step(1.0, epoch=1)  # best=1.0
        assert not es.step(1.1, epoch=2)  # counter=1
        assert es.step(1.2, epoch=3)  # counter=2 → 停止

    def test_min_mode_improvement_resets_counter(self) -> None:
        """min モードで改善があればカウンタリセット."""
        es = EarlyStopping(patience=2, mode="min")

        assert not es.step(1.0, epoch=1)
        assert not es.step(1.1, epoch=2)  # counter=1
        assert not es.step(0.8, epoch=3)  # 改善 → counter=0
        assert es.best_value == 0.8

    def test_min_delta_threshold(self) -> None:
        """min_delta 未満の変化は改善と見なさない."""
        es = EarlyStopping(patience=2, mode="max", min_delta=0.01)

        assert not es.step(0.50, epoch=1)  # best=0.50
        assert not es.step(0.505, epoch=2)  # +0.005 < min_delta → counter=1
        assert es.step(0.509, epoch=3)  # +0.009 < min_delta → counter=2 → 停止

    def test_min_delta_sufficient_improvement(self) -> None:
        """min_delta 以上の変化は改善と見なす."""
        es = EarlyStopping(patience=2, mode="max", min_delta=0.01)

        assert not es.step(0.50, epoch=1)
        assert not es.step(0.52, epoch=2)  # +0.02 > min_delta → 改善
        assert es.counter == 0
        assert es.best_value == 0.52

    def test_first_step_never_stops(self) -> None:
        """初回の step は必ず False を返す."""
        es = EarlyStopping(patience=1, mode="max")
        assert not es.step(0.0, epoch=1)
        assert es.best_value == 0.0
        assert es.best_epoch == 1

    def test_patience_1_stops_after_one_no_improvement(self) -> None:
        """patience=1 で1回改善なしなら停止."""
        es = EarlyStopping(patience=1, mode="max")

        assert not es.step(0.5, epoch=1)
        assert es.step(0.4, epoch=2)

    def test_properties_initial_state(self) -> None:
        """初期状態のプロパティ値を確認."""
        es = EarlyStopping(patience=5, mode="max")

        assert es.patience == 5
        assert es.counter == 0
        assert es.best_value is None
        assert es.best_epoch == 0

    def test_best_epoch_tracks_correctly(self) -> None:
        """best_epoch が最良メトリクスのエポックを追跡する."""
        es = EarlyStopping(patience=5, mode="max")

        es.step(0.3, epoch=1)
        es.step(0.5, epoch=2)
        es.step(0.4, epoch=3)
        es.step(0.6, epoch=4)
        es.step(0.55, epoch=5)

        assert es.best_epoch == 4
        assert es.best_value == 0.6

    def test_max_mode_equal_value_is_not_improvement(self) -> None:
        """max モードで同値は改善と見なさない."""
        es = EarlyStopping(patience=2, mode="max")

        assert not es.step(0.5, epoch=1)
        assert not es.step(0.5, epoch=2)  # 同値 → counter=1
        assert es.counter == 1

    def test_min_mode_equal_value_is_not_improvement(self) -> None:
        """min モードで同値は改善と見なさない."""
        es = EarlyStopping(patience=2, mode="min")

        assert not es.step(1.0, epoch=1)
        assert not es.step(1.0, epoch=2)  # 同値 → counter=1
        assert es.counter == 1

    def test_min_mode_min_delta_threshold(self) -> None:
        """min モードで min_delta 未満の変化は改善と見なさない."""
        es = EarlyStopping(patience=2, mode="min", min_delta=0.01)

        assert not es.step(1.00, epoch=1)  # best=1.00
        assert not es.step(0.995, epoch=2)  # -0.005 < min_delta → counter=1
        assert es.step(0.996, epoch=3)  # counter=2 → 停止

    def test_min_mode_min_delta_sufficient_improvement(self) -> None:
        """min モードで min_delta 以上の変化は改善と見なす."""
        es = EarlyStopping(patience=2, mode="min", min_delta=0.01)

        assert not es.step(1.00, epoch=1)
        assert not es.step(0.98, epoch=2)  # -0.02 > min_delta → 改善
        assert es.counter == 0
        assert es.best_value == 0.98

    def test_invalid_patience_raises_error(self) -> None:
        """patience < 1 で ValueError."""
        with pytest.raises(ValueError, match="patience は 1 以上"):
            EarlyStopping(patience=0)

    def test_negative_min_delta_raises_error(self) -> None:
        """min_delta < 0 で ValueError."""
        with pytest.raises(ValueError, match="min_delta は 0 以上"):
            EarlyStopping(patience=5, min_delta=-0.1)
