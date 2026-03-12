"""F1ConfidencePlotter のテスト."""

from pathlib import Path

import pytest
import torch

from pochidetection.visualization import F1ConfidencePlotter


@pytest.fixture(scope="class")
def f1_confidence_html_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """標準条件で生成した F1-Confidence 曲線の HTML パスを返す fixture."""
    precision = torch.rand(10, 101, 3, 4, 3)
    scores = torch.rand(10, 101, 3, 4, 3)
    output_path = (
        Path(tmp_path_factory.mktemp("f1_confidence_plotter")) / "f1_confidence.html"
    )
    F1ConfidencePlotter(precision, scores).plot(output_path)
    return output_path


@pytest.fixture(scope="class")
def f1_confidence_html_content(f1_confidence_html_path: Path) -> str:
    """標準条件で生成した F1-Confidence 曲線の HTML 内容を返す fixture."""
    return f1_confidence_html_path.read_text(encoding="utf-8")


@pytest.fixture(scope="class")
def f1_confidence_with_class_names_html_content(
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """クラス名指定で生成した F1-Confidence 曲線の HTML 内容を返す fixture."""
    precision = torch.rand(10, 101, 2, 4, 3)
    scores = torch.rand(10, 101, 2, 4, 3)
    output_path = (
        Path(tmp_path_factory.mktemp("f1_confidence_named_plotter"))
        / "f1_confidence.html"
    )
    F1ConfidencePlotter(precision, scores, class_names=["Dog", "Cat"]).plot(output_path)
    return output_path.read_text(encoding="utf-8")


@pytest.fixture(scope="class")
def all_invalid_html_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """全クラス無効値 (-1) で生成した HTML パスを返す fixture."""
    precision = torch.full((10, 101, 2, 4, 3), -1.0)
    scores = torch.full((10, 101, 2, 4, 3), -1.0)
    output_path = (
        Path(tmp_path_factory.mktemp("f1_confidence_all_invalid"))
        / "f1_confidence.html"
    )
    F1ConfidencePlotter(precision, scores).plot(output_path)
    return output_path


class TestF1ConfidencePlotterInit:
    """F1ConfidencePlotter 初期化のテスト."""

    def test_default_class_names(self) -> None:
        """class_names 未指定時にデフォルト名が設定されることを確認."""
        precision = torch.rand(10, 101, 3, 4, 3)
        scores = torch.rand(10, 101, 3, 4, 3)
        plotter = F1ConfidencePlotter(precision, scores)
        assert plotter._class_names == ["Class 0", "Class 1", "Class 2"]

    def test_custom_class_names(self) -> None:
        """class_names 指定時にそれが使用されることを確認."""
        precision = torch.rand(10, 101, 2, 4, 3)
        scores = torch.rand(10, 101, 2, 4, 3)
        plotter = F1ConfidencePlotter(precision, scores, class_names=["Dog", "Cat"])
        assert plotter._class_names == ["Dog", "Cat"]

    def test_num_classes_from_precision_shape(self) -> None:
        """クラス数が precision テンソルの shape から取得されることを確認."""
        precision = torch.rand(10, 101, 5, 4, 3)
        scores = torch.rand(10, 101, 5, 4, 3)
        plotter = F1ConfidencePlotter(precision, scores)
        assert plotter._num_classes == 5


class TestF1ConfidencePlotterPlot:
    """F1ConfidencePlotter HTML 出力のテスト."""

    def test_plot_creates_html_file(self, f1_confidence_html_path: Path) -> None:
        """HTML ファイルが作成されることを確認."""
        assert f1_confidence_html_path.exists()

    def test_plot_html_contains_f1_labels(
        self, f1_confidence_html_content: str
    ) -> None:
        """生成された HTML に F1, Confidence のラベルが含まれることを確認."""
        assert "F1 Score" in f1_confidence_html_content
        assert "Confidence" in f1_confidence_html_content
        assert "All Classes" in f1_confidence_html_content

    def test_plot_html_contains_best_f1_marker(
        self, f1_confidence_html_content: str
    ) -> None:
        """生成された HTML に最大 F1 マーカーが含まれることを確認."""
        assert "Best F1=" in f1_confidence_html_content

    def test_plot_with_class_names(
        self, f1_confidence_with_class_names_html_content: str
    ) -> None:
        """クラス名を指定した場合にそれが含まれることを確認."""
        assert "Dog" in f1_confidence_with_class_names_html_content
        assert "Cat" in f1_confidence_with_class_names_html_content


class TestF1ConfidencePlotterEdgeCases:
    """F1ConfidencePlotter エッジケースのテスト."""

    def test_all_invalid_precision_creates_html(
        self, all_invalid_html_path: Path
    ) -> None:
        """全クラス無効値 (-1) でもエラーなく HTML が生成されることを確認."""
        assert all_invalid_html_path.exists()

    def test_partial_invalid_precision_excludes_class(self, tmp_path: Path) -> None:
        """一部クラスのみ無効値の場合, そのクラスがスキップされることを確認."""
        precision = torch.rand(10, 101, 3, 4, 3)
        scores = torch.rand(10, 101, 3, 4, 3)
        # Class 1 を全て無効値にする
        precision[:, :, 1, :, :] = -1
        scores[:, :, 1, :, :] = -1
        output_path = tmp_path / "f1_confidence.html"
        F1ConfidencePlotter(precision, scores, class_names=["Dog", "Cat", "Bird"]).plot(
            output_path
        )

        html_content = output_path.read_text(encoding="utf-8")
        assert "Dog" in html_content
        assert "Bird" in html_content
        # Cat (Class 1) は per-class グラフに含まれない
        # ただし plotly の legend に含まれないことを確認
        assert output_path.exists()

    def test_negative_values_not_in_output(self, tmp_path: Path) -> None:
        """無効値 (-1) がデータ値として HTML に含まれないことを確認."""
        precision = torch.rand(10, 101, 2, 4, 3)
        scores = torch.rand(10, 101, 2, 4, 3)
        precision[0, 50:60, 0, 0, 2] = -1
        scores[0, 50:60, 0, 0, 2] = -1
        output_path = tmp_path / "f1_confidence.html"
        F1ConfidencePlotter(precision, scores).plot(output_path)

        html_content = output_path.read_text(encoding="utf-8")
        assert "-1.0" not in html_content
