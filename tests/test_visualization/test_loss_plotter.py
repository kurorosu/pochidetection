"""LossPlotter のテスト."""

from pathlib import Path

from pochidetection.utils import TrainingHistory
from pochidetection.visualization import LossPlotter


class TestLossPlotter:
    """LossPlotter クラスのテスト."""

    def test_plot_creates_html_file(
        self, tmp_path: Path, training_history: TrainingHistory
    ) -> None:
        """HTML ファイルが作成されることを確認."""
        output_path = tmp_path / "loss_curve.html"
        plotter = LossPlotter(training_history)
        plotter.plot(output_path)

        assert output_path.exists()

    def test_plot_html_contains_plotly(
        self, tmp_path: Path, single_epoch_history: TrainingHistory
    ) -> None:
        """生成された HTML に plotly のコンテンツが含まれることを確認."""
        output_path = tmp_path / "loss_curve.html"
        plotter = LossPlotter(single_epoch_history)
        plotter.plot(output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "plotly" in content.lower()
        assert "Train Loss" in content
        assert "Val Loss" in content
