"""PRCurvePlotter のテスト."""

from pathlib import Path

import torch

from pochidetection.visualization import PRCurvePlotter


class TestPRCurvePlotter:
    """PRCurvePlotter クラスのテスト."""

    def test_plot_creates_html_file(self, tmp_path: Path) -> None:
        """HTML ファイルが作成されることを確認."""
        # ダミーのprecisionテンソルを作成
        # 形状: (T, R, K, A, M) = (10, 101, 3, 4, 3)
        # T=IoU閾値数, R=Recall閾値数, K=クラス数, A=領域数, M=最大検出数
        precision = torch.rand(10, 101, 3, 4, 3)

        output_path = tmp_path / "pr_curve.html"
        plotter = PRCurvePlotter(precision)
        plotter.plot(output_path)

        assert output_path.exists()

    def test_plot_html_contains_pr_labels(self, tmp_path: Path) -> None:
        """生成された HTML に Precision, Recall のラベルが含まれることを確認."""
        precision = torch.rand(10, 101, 2, 4, 3)

        output_path = tmp_path / "pr_curve.html"
        plotter = PRCurvePlotter(precision)
        plotter.plot(output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "Precision" in content
        assert "Recall" in content
        assert "All Classes" in content

    def test_plot_with_class_names(self, tmp_path: Path) -> None:
        """クラス名を指定した場合にそれが含まれることを確認."""
        precision = torch.rand(10, 101, 2, 4, 3)
        class_names = ["Dog", "Cat"]

        output_path = tmp_path / "pr_curve.html"
        plotter = PRCurvePlotter(precision, class_names=class_names)
        plotter.plot(output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "Dog" in content
        assert "Cat" in content

    def test_plot_handles_invalid_precision_values(self, tmp_path: Path) -> None:
        """無効値(-1)を含むprecisionテンソルを処理できることを確認."""
        precision = torch.rand(10, 101, 2, 4, 3)
        # 一部を無効値に設定
        precision[:, :, 1, :, :] = -1

        output_path = tmp_path / "pr_curve.html"
        plotter = PRCurvePlotter(precision)
        plotter.plot(output_path)

        assert output_path.exists()
