"""MetricsPlotter のテスト."""

from pathlib import Path

from pochidetection.utils import TrainingHistory
from pochidetection.visualization import MetricsPlotter


class TestMetricsPlotter:
    """MetricsPlotter クラスのテスト."""

    def test_plot_creates_html_file(
        self, tmp_path: Path, training_history: TrainingHistory
    ) -> None:
        """HTML ファイルが作成されることを確認."""
        output_path = tmp_path / "metrics_curve.html"
        plotter = MetricsPlotter(training_history)
        plotter.plot(output_path)

        assert output_path.exists()

    def test_plot_html_contains_map_labels(
        self, tmp_path: Path, single_epoch_history: TrainingHistory
    ) -> None:
        """生成された HTML に mAP のラベルが含まれることを確認."""
        output_path = tmp_path / "metrics_curve.html"
        plotter = MetricsPlotter(single_epoch_history)
        plotter.plot(output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "mAP" in content
        assert "mAP@50" in content
        assert "mAP@75" in content
