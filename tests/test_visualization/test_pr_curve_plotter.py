"""PRCurvePlotter のテスト."""

from pathlib import Path

import pytest
import torch

from pochidetection.visualization import PRCurvePlotter


@pytest.fixture(scope="class")
def pr_curve_html_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """標準条件で生成した PR 曲線の HTML パスを返すfixture."""
    precision = torch.rand(10, 101, 3, 4, 3)
    output_path = Path(tmp_path_factory.mktemp("pr_curve_plotter")) / "pr_curve.html"
    PRCurvePlotter(precision).plot(output_path)
    return output_path


@pytest.fixture(scope="class")
def pr_curve_html_content(pr_curve_html_path: Path) -> str:
    """標準条件で生成した PR 曲線の HTML 内容を返すfixture."""
    return pr_curve_html_path.read_text(encoding="utf-8")


@pytest.fixture(scope="class")
def pr_curve_with_class_names_html_content(
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """クラス名指定で生成した PR 曲線の HTML 内容を返すfixture."""
    precision = torch.rand(10, 101, 2, 4, 3)
    output_path = (
        Path(tmp_path_factory.mktemp("pr_curve_named_plotter")) / "pr_curve.html"
    )
    PRCurvePlotter(precision, class_names=["Dog", "Cat"]).plot(output_path)
    return output_path.read_text(encoding="utf-8")


@pytest.fixture(scope="class")
def invalid_pr_curve_html_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """無効値を含む precision で生成した PR 曲線の HTML パスを返すfixture."""
    precision = torch.rand(10, 101, 2, 4, 3)
    precision[:, :, 1, :, :] = -1
    output_path = (
        Path(tmp_path_factory.mktemp("pr_curve_invalid_plotter")) / "pr_curve.html"
    )
    PRCurvePlotter(precision).plot(output_path)
    return output_path


class TestPRCurvePlotter:
    """PRCurvePlotter クラスのテスト."""

    def test_plot_creates_html_file(self, pr_curve_html_path: Path) -> None:
        """HTML ファイルが作成されることを確認."""
        assert pr_curve_html_path.exists()

    def test_plot_html_contains_pr_labels(self, pr_curve_html_content: str) -> None:
        """生成された HTML に Precision, Recall のラベルが含まれることを確認."""
        assert "Precision" in pr_curve_html_content
        assert "Recall" in pr_curve_html_content
        assert "All Classes" in pr_curve_html_content

    def test_plot_with_class_names(
        self, pr_curve_with_class_names_html_content: str
    ) -> None:
        """クラス名を指定した場合にそれが含まれることを確認."""
        assert "Dog" in pr_curve_with_class_names_html_content
        assert "Cat" in pr_curve_with_class_names_html_content

    def test_plot_handles_invalid_precision_values(
        self, invalid_pr_curve_html_path: Path
    ) -> None:
        """無効値(-1)を含むprecisionテンソルを処理できることを確認."""
        assert invalid_pr_curve_html_path.exists()

    def test_partial_invalid_precision_excludes_negative_values(
        self, tmp_path: Path
    ) -> None:
        """一部の recall ポイントのみ -1 の場合, -1 が描画データに含まれないことを確認."""
        precision = torch.rand(10, 101, 2, 4, 3)
        # Class 0 の一部の recall ポイントを -1 に設定
        precision[0, 50:60, 0, 0, 2] = -1
        output_path = tmp_path / "pr_curve.html"
        PRCurvePlotter(precision).plot(output_path)

        html_content = output_path.read_text(encoding="utf-8")
        assert output_path.exists()
        # -1 がデータ値として含まれていないことを確認
        # plotly は NaN を "null" として出力する
        assert "-1.0" not in html_content
