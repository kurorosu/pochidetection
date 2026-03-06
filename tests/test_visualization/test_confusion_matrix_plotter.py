"""confusion_matrix_plotter のテスト."""

import json
from pathlib import Path

import pytest
import torch

from pochidetection.core.detection import Detection
from pochidetection.visualization import ConfusionMatrixPlotter, build_confusion_matrix


@pytest.fixture()
def class_names() -> list[str]:
    """テスト用クラス名."""
    return ["cat", "dog"]


class TestBuildConfusionMatrix:
    """build_confusion_matrix のテスト."""

    def test_matrix_shape(
        self,
        sample_predictions: dict[str, list[Detection]],
        coco_annotation: Path,
        class_names: list[str],
    ) -> None:
        """行列の形状が (num_classes+1) x (num_classes+1)."""
        matrix = build_confusion_matrix(
            sample_predictions, coco_annotation, class_names
        )
        assert matrix.shape == (3, 3)

    def test_matrix_dtype(
        self,
        sample_predictions: dict[str, list[Detection]],
        coco_annotation: Path,
        class_names: list[str],
    ) -> None:
        """行列の dtype が int64."""
        matrix = build_confusion_matrix(
            sample_predictions, coco_annotation, class_names
        )
        assert matrix.dtype == torch.int64

    def test_tp_on_diagonal(
        self,
        sample_predictions: dict[str, list[Detection]],
        coco_annotation: Path,
        class_names: list[str],
    ) -> None:
        """TP は対角要素に計上される."""
        matrix = build_confusion_matrix(
            sample_predictions, coco_annotation, class_names
        )
        # img001: cat TP, dog TP / img002: cat TP → cat=2, dog=1
        assert matrix[0, 0] == 2
        assert matrix[1, 1] == 1

    def test_fn_in_background_column(
        self,
        sample_predictions: dict[str, list[Detection]],
        coco_annotation: Path,
        class_names: list[str],
    ) -> None:
        """検出漏れ (FN) は Background 列に計上される."""
        matrix = build_confusion_matrix(
            sample_predictions, coco_annotation, class_names
        )
        # img003 は検出なし, GT に cat 1 つ -> matrix[0, 2] (cat → Background) = 1
        assert matrix[0, 2] == 1
        assert matrix[1, 2] == 0

    def test_fp_in_background_row(
        self,
        class_names: list[str],
        coco_annotation: Path,
    ) -> None:
        """誤検出 (FP) は Background 行に計上される."""
        predictions: dict[str, list[Detection]] = {
            "img001.jpg": [
                Detection(box=[10.0, 20.0, 100.0, 200.0], score=0.95, label=0),
                Detection(box=[50.0, 60.0, 150.0, 250.0], score=0.80, label=1),
                # GT にマッチしない検出
                Detection(box=[400.0, 400.0, 500.0, 500.0], score=0.60, label=0),
            ],
        }
        matrix = build_confusion_matrix(predictions, coco_annotation, class_names)
        # img001: 2 TP + 1 FP(cat) / img002,img003: 各 GT が FN
        bg_row = matrix[2, :]
        assert bg_row[0] == 1  # FP cat
        assert bg_row[1] == 0
        assert bg_row[2] == 0

    def test_no_detections_all_fn(
        self,
        coco_annotation: Path,
        class_names: list[str],
    ) -> None:
        """全画像で検出なしの場合, 全 GT が FN になる."""
        predictions: dict[str, list[Detection]] = {
            "img001.jpg": [],
            "img002.jpg": [],
            "img003.jpg": [],
        }
        matrix = build_confusion_matrix(predictions, coco_annotation, class_names)
        # 4 つの GT が全て FN: cat 3つ, dog 1つ
        assert matrix[0, 2] == 3  # cat → Background
        assert matrix[1, 2] == 1  # dog → Background

    def test_total_count_matches(
        self,
        sample_predictions: dict[str, list[Detection]],
        coco_annotation: Path,
        class_names: list[str],
    ) -> None:
        """行列の合計値が検出数 + FN 数と一致する."""
        matrix = build_confusion_matrix(
            sample_predictions, coco_annotation, class_names
        )
        total = matrix.sum().item()
        # 検出 3 (TP: cat=2, dog=1) + FN 1 (img003 の cat) = 4
        assert total == 4

    def test_single_class(self, tmp_path: Path) -> None:
        """1 クラスの場合, 2x2 行列になる."""
        ann = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            ],
            "categories": [{"id": 1, "name": "cat"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50]},
            ],
        }
        ann_path = tmp_path / "ann.json"
        ann_path.write_text(json.dumps(ann), encoding="utf-8")

        predictions: dict[str, list[Detection]] = {
            "a.jpg": [Detection(box=[0, 0, 50, 50], score=0.9, label=0)],
        }
        matrix = build_confusion_matrix(predictions, ann_path, ["cat"])
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 1  # TP


class TestConfusionMatrixPlotter:
    """ConfusionMatrixPlotter のテスト."""

    def test_plot_creates_html_file(self, tmp_path: Path) -> None:
        """HTML ファイルが作成される."""
        matrix = torch.tensor([[5, 1, 0], [2, 8, 1], [1, 0, 0]], dtype=torch.int64)
        plotter = ConfusionMatrixPlotter(matrix, ["cat", "dog"])
        output_path = tmp_path / "cm.html"
        plotter.plot(output_path)
        assert output_path.exists()

    def test_plot_html_contains_plotly(self, tmp_path: Path) -> None:
        """HTML に Plotly スクリプトが含まれる."""
        matrix = torch.tensor([[3, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=torch.int64)
        plotter = ConfusionMatrixPlotter(matrix, ["cat", "dog"])
        output_path = tmp_path / "cm.html"
        plotter.plot(output_path)
        html = output_path.read_text(encoding="utf-8")
        assert "plotly" in html.lower()

    def test_plot_html_contains_class_names(self, tmp_path: Path) -> None:
        """HTML にクラス名が含まれる."""
        matrix = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.int64)
        plotter = ConfusionMatrixPlotter(matrix, ["cat", "dog"])
        output_path = tmp_path / "cm.html"
        plotter.plot(output_path)
        html = output_path.read_text(encoding="utf-8")
        assert "cat" in html
        assert "dog" in html
        assert "Background" in html

    def test_plot_html_contains_title(self, tmp_path: Path) -> None:
        """HTML にタイトルが含まれる."""
        matrix = torch.zeros(2, 2, dtype=torch.int64)
        plotter = ConfusionMatrixPlotter(matrix, ["cat"])
        output_path = tmp_path / "cm.html"
        plotter.plot(output_path)
        html = output_path.read_text(encoding="utf-8")
        assert "Confusion Matrix" in html
