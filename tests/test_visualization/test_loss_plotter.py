"""LossPlotter のテスト."""

from pathlib import Path

import pytest

from pochidetection.utils import TrainingHistory
from pochidetection.visualization import LossPlotter


@pytest.fixture(scope="class")
def loss_plot_html_path(
    tmp_path_factory: pytest.TempPathFactory,
    single_epoch_history: TrainingHistory,
) -> Path:
    """LossPlotter の出力 HTML パスを返すfixture."""
    output_path = Path(tmp_path_factory.mktemp("loss_plotter")) / "loss_curve.html"
    LossPlotter(single_epoch_history).plot(output_path)
    return output_path


@pytest.fixture(scope="class")
def loss_plot_html_content(loss_plot_html_path: Path) -> str:
    """LossPlotter の出力 HTML を返すfixture."""
    return loss_plot_html_path.read_text(encoding="utf-8")


class TestLossPlotter:
    """LossPlotter クラスのテスト."""

    def test_plot_creates_html_file(self, loss_plot_html_path: Path) -> None:
        """HTML ファイルが作成されることを確認."""
        assert loss_plot_html_path.exists()

    def test_plot_html_contains_plotly(self, loss_plot_html_content: str) -> None:
        """生成された HTML に plotly のコンテンツが含まれることを確認."""
        assert "plotly" in loss_plot_html_content.lower()
        assert "Train Loss" in loss_plot_html_content
        assert "Val Loss" in loss_plot_html_content
