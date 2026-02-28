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

    def test_plot_creates_html_file(self, tmp_path: Path) -> None:
        """HTML ファイルが作成されることを確認."""
        history = TrainingHistory()
        history.add(1, 0.5, 0.4, 0.3, 0.5, 0.2, 0.001)
        history.add(2, 0.4, 0.3, 0.4, 0.6, 0.3, 0.001)
        history.add(3, 0.3, 0.25, 0.5, 0.7, 0.4, 0.0005)

        output_path = tmp_path / "training_report.html"
        loss_plotter = LossPlotter(history)
        metrics_plotter = MetricsPlotter(history)
        plotter = TrainingReportPlotter(loss_plotter, metrics_plotter)
        plotter.plot(output_path)

        assert output_path.exists()

    def test_plot_html_contains_both_charts(self, tmp_path: Path) -> None:
        """生成された HTML に Loss と mAP のラベルが含まれることを確認."""
        history = TrainingHistory()
        history.add(1, 0.5, 0.4, 0.3, 0.5, 0.2, 0.001)

        output_path = tmp_path / "training_report.html"
        loss_plotter = LossPlotter(history)
        metrics_plotter = MetricsPlotter(history)
        plotter = TrainingReportPlotter(loss_plotter, metrics_plotter)
        plotter.plot(output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "Train Loss" in content
        assert "Val Loss" in content
        assert "mAP" in content
        assert "mAP@50" in content
