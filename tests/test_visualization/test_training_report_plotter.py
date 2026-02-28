"""TrainingReportPlotter のテスト."""

from pathlib import Path

import pytest

from pochidetection.utils import TrainingHistory
from pochidetection.visualization import (
    LossPlotter,
    MetricsPlotter,
    TrainingReportPlotter,
)


@pytest.fixture(scope="class")
def training_report_html_path(
    tmp_path_factory: pytest.TempPathFactory,
    single_epoch_history: TrainingHistory,
) -> Path:
    """TrainingReportPlotter の出力 HTML パスを返すfixture."""
    output_path = (
        Path(tmp_path_factory.mktemp("training_report_plotter"))
        / "training_report.html"
    )
    loss_plotter = LossPlotter(single_epoch_history)
    metrics_plotter = MetricsPlotter(single_epoch_history)
    TrainingReportPlotter(loss_plotter, metrics_plotter).plot(output_path)
    return output_path


@pytest.fixture(scope="class")
def training_report_html_content(training_report_html_path: Path) -> str:
    """TrainingReportPlotter の出力 HTML を返すfixture."""
    return training_report_html_path.read_text(encoding="utf-8")


class TestTrainingReportPlotter:
    """TrainingReportPlotter クラスのテスト."""

    def test_plot_creates_html_file(self, training_report_html_path: Path) -> None:
        """HTML ファイルが作成されることを確認."""
        assert training_report_html_path.exists()

    def test_plot_html_contains_both_charts(
        self, training_report_html_content: str
    ) -> None:
        """生成された HTML に Loss と mAP のラベルが含まれることを確認."""
        assert "Train Loss" in training_report_html_content
        assert "Val Loss" in training_report_html_content
        assert "mAP" in training_report_html_content
        assert "mAP@50" in training_report_html_content
