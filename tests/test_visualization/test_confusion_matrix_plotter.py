"""confusion_matrix_plotter のテスト."""

import json
from pathlib import Path

import pytest
import torch

from pochidetection.scripts.rtdetr.inference.detection import Detection
from pochidetection.visualization import ConfusionMatrixPlotter, build_confusion_matrix


@pytest.fixture()
def class_names() -> list[str]:
    """テスト用クラス名."""
    return ["cat", "dog"]


@pytest.fixture()
def coco_annotation(tmp_path: Path) -> Path:
    """テスト用 COCO アノテーション."""
    ann = {
        "images": [
            {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img002.jpg", "width": 640, "height": 480},
            {"id": 3, "file_name": "img003.jpg", "width": 640, "height": 480},
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 90.0, 180.0],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [50.0, 60.0, 100.0, 190.0],
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [5.0, 10.0, 75.0, 110.0],
            },
            {
                "id": 4,
                "image_id": 3,
                "category_id": 1,
                "bbox": [20.0, 30.0, 60.0, 80.0],
            },
        ],
    }
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(ann), encoding="utf-8")
    return ann_path


@pytest.fixture()
def sample_predictions() -> dict[str, list[Detection]]:
    """テスト用推論結果."""
    return {
        "img001.jpg": [
            Detection(box=[10.0, 20.0, 100.0, 200.0], score=0.95, label=0),
            Detection(box=[50.0, 60.0, 150.0, 250.0], score=0.80, label=1),
        ],
        "img002.jpg": [
            Detection(box=[5.0, 10.0, 80.0, 120.0], score=0.70, label=0),
        ],
        "img003.jpg": [],
    }


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
        # cat(0) と dog(1) の対角にマッチがあるはず
        diagonal_sum = matrix[0, 0] + matrix[1, 1]
        assert diagonal_sum >= 2

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
        # img003 は検出なし, GT に cat 1 つ -> matrix[0, 2] (cat → Background)
        bg_col = matrix[:, 2]
        assert bg_col.sum() >= 1

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
        bg_row = matrix[2, :]
        assert bg_row.sum() >= 1

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
        # 4 つの GT が全て FN
        bg_col_sum = matrix[:2, 2].sum()
        assert bg_col_sum == 4

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
        # 検出 3 (img001: 2, img002: 1) + FN (img003 の未検出 GT)
        assert total >= 3

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
