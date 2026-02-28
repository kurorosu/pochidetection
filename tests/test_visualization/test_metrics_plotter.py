"""MetricsPlotter のテスト."""

from pathlib import Path

import pytest

from pochidetection.utils import TrainingHistory
from pochidetection.visualization import MetricsPlotter


@pytest.fixture(scope="class")
def metrics_plot_html_path(
    tmp_path_factory: pytest.TempPathFactory,
    single_epoch_history: TrainingHistory,
) -> Path:
    """MetricsPlotter の出力 HTML パスを返すfixture."""
    output_path = (
        Path(tmp_path_factory.mktemp("metrics_plotter")) / "metrics_curve.html"
    )
    MetricsPlotter(single_epoch_history).plot(output_path)
    return output_path


@pytest.fixture(scope="class")
def metrics_plot_html_content(metrics_plot_html_path: Path) -> str:
    """MetricsPlotter の出力 HTML を返すfixture."""
    return metrics_plot_html_path.read_text(encoding="utf-8")


class TestMetricsPlotter:
    """MetricsPlotter クラスのテスト."""

    def test_plot_creates_html_file(self, metrics_plot_html_path: Path) -> None:
        """HTML ファイルが作成されることを確認."""
        assert metrics_plot_html_path.exists()

    def test_plot_html_contains_map_labels(
        self, metrics_plot_html_content: str
    ) -> None:
        """生成された HTML に mAP のラベルが含まれることを確認."""
        assert "mAP" in metrics_plot_html_content
        assert "mAP@50" in metrics_plot_html_content
        assert "mAP@75" in metrics_plot_html_content
