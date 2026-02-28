"""TrainingReportPlotter のテスト."""

from pathlib import Path

from pochidetection.utils import TrainingHistory
from pochidetection.visualization import (
    LossPlotter,
    MetricsPlotter,
    TrainingReportPlotter,
)


class TestTrainingReportPlotter:
    """TrainingReportPlotter クラスのテスト."""

    def test_plot_creates_html_file(
        self, tmp_path: Path, training_history: TrainingHistory
    ) -> None:
        """HTML ファイルが作成されることを確認."""
        output_path = tmp_path / "training_report.html"
        loss_plotter = LossPlotter(training_history)
        metrics_plotter = MetricsPlotter(training_history)
        plotter = TrainingReportPlotter(loss_plotter, metrics_plotter)
        plotter.plot(output_path)

        assert output_path.exists()

    def test_plot_html_contains_both_charts(
        self, tmp_path: Path, single_epoch_history: TrainingHistory
    ) -> None:
        """生成された HTML に Loss と mAP のラベルが含まれることを確認."""
        output_path = tmp_path / "training_report.html"
        loss_plotter = LossPlotter(single_epoch_history)
        metrics_plotter = MetricsPlotter(single_epoch_history)
        plotter = TrainingReportPlotter(loss_plotter, metrics_plotter)
        plotter.plot(output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "Train Loss" in content
        assert "Val Loss" in content
        assert "mAP" in content
        assert "mAP@50" in content
