"""SSD 系モデル共通の BN 退避・復元ヘルパーのテスト."""

import torch
import torch.nn as nn

from pochidetection.scripts.ssd.validation import restore_bn_states, save_bn_states


class TestSaveBnStates:
    """save_bn_states のテスト."""

    def test_saves_running_mean_and_var(self) -> None:
        """running_mean, running_var, num_batches_tracked が保存される."""
        model = nn.Sequential(nn.BatchNorm2d(3))
        # ダミーデータで forward して running statistics を更新
        model.train()
        model(torch.randn(2, 3, 4, 4))

        states = save_bn_states(model)

        assert "0.running_mean" in states
        assert "0.running_var" in states
        assert "0.num_batches_tracked" in states

    def test_saved_states_are_clones(self) -> None:
        """保存された値が元テンソルと独立したクローンである."""
        model = nn.Sequential(nn.BatchNorm2d(3))
        model.train()
        model(torch.randn(2, 3, 4, 4))

        states = save_bn_states(model)
        original_mean = states["0.running_mean"].clone()

        # モデルの running_mean を変更
        model[0].running_mean.fill_(999.0)  # type: ignore[union-attr]

        # 保存された値は変わらない
        assert torch.equal(states["0.running_mean"], original_mean)

    def test_no_bn_returns_empty(self) -> None:
        """BatchNorm がないモデルでは空辞書を返す."""
        model = nn.Sequential(nn.Linear(3, 3))
        states = save_bn_states(model)
        assert states == {}


class TestRestoreBnStates:
    """restore_bn_states のテスト."""

    def test_restores_after_forward(self) -> None:
        """forward による変更後も退避した値が復元される."""
        model = nn.Sequential(nn.BatchNorm2d(3))
        model.train()
        model(torch.randn(2, 3, 4, 4))

        # 退避
        states = save_bn_states(model)
        saved_mean = states["0.running_mean"].clone()
        saved_var = states["0.running_var"].clone()

        # 追加の forward で running statistics を変更
        model(torch.randn(2, 3, 4, 4))
        model(torch.randn(2, 3, 4, 4))

        # 復元
        restore_bn_states(model, states)

        assert torch.equal(model[0].running_mean, saved_mean)  # type: ignore[arg-type]
        assert torch.equal(model[0].running_var, saved_var)  # type: ignore[arg-type]

    def test_restore_with_empty_states(self) -> None:
        """空の states で呼んでもエラーにならない."""
        model = nn.Sequential(nn.BatchNorm2d(3))
        model.train()
        model(torch.randn(2, 3, 4, 4))

        original_mean = model[0].running_mean.clone()  # type: ignore[union-attr]
        restore_bn_states(model, {})

        # 変更されない
        assert torch.equal(model[0].running_mean, original_mean)  # type: ignore[arg-type]


class TestBnStatesRoundTrip:
    """save → train forward → restore の統合テスト."""

    def test_validation_does_not_contaminate_bn(self) -> None:
        """validation ループを模擬し, BN 統計が保護されることを確認."""
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        # 学習フェーズ: 統計を蓄積
        model.train()
        for _ in range(5):
            model(torch.randn(4, 3, 8, 8))

        # 学習後の BN 統計を記録
        trained_mean = model[1].running_mean.clone()  # type: ignore[union-attr]
        trained_var = model[1].running_var.clone()  # type: ignore[union-attr]

        # validation フェーズ: 退避 → train() forward → 復元
        bn_states = save_bn_states(model)

        model.eval()
        with torch.no_grad():
            # 分布の異なるデータで train モード forward (loss 計算を模擬)
            model.train()
            model(torch.randn(4, 3, 8, 8) * 10 + 5)
            model.eval()

        restore_bn_states(model, bn_states)

        # BN 統計が学習時の値に戻っている
        assert torch.equal(model[1].running_mean, trained_mean)  # type: ignore[arg-type]
        assert torch.equal(model[1].running_var, trained_var)  # type: ignore[arg-type]
