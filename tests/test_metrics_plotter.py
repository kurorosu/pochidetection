"""MetricsPlotter のテスト."""

from pathlib import Path

from pochidetection.utils import TrainingHistory
from pochidetection.visualization import MetricsPlotter


class TestMetricsPlotter:
    """MetricsPlotter クラスのテスト."""

    def test_plot_creates_html_file(self, tmp_path: Path) -> None:
        """HTML ファイルが作成されることを確認."""
        history = TrainingHistory()
        history.add(1, 0.5, 0.4, 0.3, 0.5, 0.2, 0.001)
        history.add(2, 0.4, 0.3, 0.4, 0.6, 0.3, 0.001)
        history.add(3, 0.3, 0.25, 0.5, 0.7, 0.4, 0.0005)

        output_path = tmp_path / "metrics_curve.html"
        plotter = MetricsPlotter(history)
        plotter.plot(output_path)

        assert output_path.exists()

    def test_plot_html_contains_map_labels(self, tmp_path: Path) -> None:
        """生成された HTML に mAP のラベルが含まれることを確認."""
        history = TrainingHistory()
        history.add(1, 0.5, 0.4, 0.3, 0.5, 0.2, 0.001)

        output_path = tmp_path / "metrics_curve.html"
        plotter = MetricsPlotter(history)
        plotter.plot(output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "mAP" in content
        assert "mAP@50" in content
        assert "mAP@75" in content
