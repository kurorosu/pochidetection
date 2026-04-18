"""detection_results_writer のテスト."""

import csv
from pathlib import Path

import pytest

from pochidetection.core.detection import Detection
from pochidetection.reporting.detection_results_writer import (
    CSV_COLUMNS,
    DetectionResultRow,
    build_detection_results,
    write_detection_results_csv,
)
from pochidetection.visualization import LabelMapper


@pytest.fixture()
def label_mapper() -> LabelMapper:
    """テスト用ラベルマッパー."""
    return LabelMapper(class_names=["cat", "dog"])


class TestBuildDetectionResultsWithoutAnnotation:
    """アノテーションなしの場合のテスト."""

    def test_returns_rows_for_all_detections(
        self,
        sample_predictions: dict[str, list[Detection]],
        label_mapper: LabelMapper,
    ) -> None:
        """全検出に対して行が生成される."""
        rows = build_detection_results(sample_predictions, label_mapper)
        # img001: 2, img002: 1, img003: 0
        assert len(rows) == 3

    def test_status_is_empty_without_annotation(
        self,
        sample_predictions: dict[str, list[Detection]],
        label_mapper: LabelMapper,
    ) -> None:
        """アノテーションなしでは status が空文字."""
        rows = build_detection_results(sample_predictions, label_mapper)
        for row in rows:
            assert row.status == ""
            assert row.iou == ""
            assert row.gt_class_name == ""

    def test_detection_id_is_1_indexed(
        self,
        sample_predictions: dict[str, list[Detection]],
        label_mapper: LabelMapper,
    ) -> None:
        """Detection_id が 1-indexed."""
        rows = build_detection_results(sample_predictions, label_mapper)
        img001_rows = [r for r in rows if r.image_name == "img001.jpg"]
        assert img001_rows[0].detection_id == 1
        assert img001_rows[1].detection_id == 2

    def test_class_name_uses_label_mapper(
        self,
        sample_predictions: dict[str, list[Detection]],
        label_mapper: LabelMapper,
    ) -> None:
        """ラベルマッパーでクラス名が設定される."""
        rows = build_detection_results(sample_predictions, label_mapper)
        img001_rows = [r for r in rows if r.image_name == "img001.jpg"]
        assert img001_rows[0].class_name == "cat"
        assert img001_rows[1].class_name == "dog"

    def test_rows_sorted_by_image_name(
        self,
        sample_predictions: dict[str, list[Detection]],
        label_mapper: LabelMapper,
    ) -> None:
        """行が画像名でソートされている."""
        rows = build_detection_results(sample_predictions, label_mapper)
        image_names = [r.image_name for r in rows]
        assert image_names == sorted(image_names)


class TestBuildDetectionResultsWithAnnotation:
    """アノテーションありの場合のテスト."""

    def test_tp_detection(
        self,
        sample_predictions: dict[str, list[Detection]],
        label_mapper: LabelMapper,
        coco_annotation: Path,
    ) -> None:
        """GT とマッチする検出が TP になる."""
        rows = build_detection_results(
            sample_predictions, label_mapper, annotation_path=coco_annotation
        )
        tp_rows = [r for r in rows if r.status == "TP"]
        # img001: cat TP, dog TP / img002: cat TP → 3 TP
        assert len(tp_rows) == 3

    def test_fn_rows_for_undetected_gt(
        self,
        sample_predictions: dict[str, list[Detection]],
        label_mapper: LabelMapper,
        coco_annotation: Path,
    ) -> None:
        """検出漏れの GT が FN として出力される."""
        rows = build_detection_results(
            sample_predictions, label_mapper, annotation_path=coco_annotation
        )
        # img003 は検出なし, GT に 1 つ cat がある -> FN 1 行
        fn_rows = [r for r in rows if r.image_name == "img003.jpg" and r.status == "FN"]
        assert len(fn_rows) == 1
        assert fn_rows[0].gt_class_name == "cat"
        assert fn_rows[0].detection_id == 0
        assert fn_rows[0].class_name == ""

    def test_iou_is_float_for_tp(
        self,
        sample_predictions: dict[str, list[Detection]],
        label_mapper: LabelMapper,
        coco_annotation: Path,
    ) -> None:
        """TP の iou が float."""
        rows = build_detection_results(
            sample_predictions, label_mapper, annotation_path=coco_annotation
        )
        tp_rows = [r for r in rows if r.status == "TP"]
        for row in tp_rows:
            assert isinstance(row.iou, float)
            # 完全一致のボックスなので IoU は 1.0
            assert row.iou == pytest.approx(1.0, abs=1e-3)

    def test_total_rows_includes_fn(
        self,
        sample_predictions: dict[str, list[Detection]],
        label_mapper: LabelMapper,
        coco_annotation: Path,
    ) -> None:
        """FN 行を含めた合計行数が正しい."""
        rows = build_detection_results(
            sample_predictions, label_mapper, annotation_path=coco_annotation
        )
        # 検出: 3 (img001: 2, img002: 1) + FN: 1 (img003)
        det_rows = [r for r in rows if r.status != "FN"]
        fn_rows = [r for r in rows if r.status == "FN"]
        assert len(det_rows) == 3
        assert len(fn_rows) == 1


class TestWriteDetectionResultsCsv:
    """CSV 書き出しのテスト."""

    def test_writes_csv_file(self, tmp_path: Path) -> None:
        """CSV ファイルが作成される."""
        rows = [
            DetectionResultRow(
                image_name="test.jpg",
                detection_id=1,
                class_name="cat",
                confidence=0.95,
                x_min=10.0,
                y_min=20.0,
                x_max=100.0,
                y_max=200.0,
                status="TP",
                iou=0.85,
                gt_class_name="cat",
            )
        ]
        csv_path = write_detection_results_csv(tmp_path, rows)

        assert csv_path.exists()
        assert csv_path.name == "detection_results.csv"

    def test_csv_has_correct_headers(self, tmp_path: Path) -> None:
        """CSV ヘッダーが正しい."""
        rows = [
            DetectionResultRow(
                image_name="test.jpg",
                detection_id=1,
                class_name="cat",
                confidence=0.95,
                x_min=10.0,
                y_min=20.0,
                x_max=100.0,
                y_max=200.0,
                status="",
                iou="",
                gt_class_name="",
            )
        ]
        csv_path = write_detection_results_csv(tmp_path, rows)

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == CSV_COLUMNS

    def test_csv_row_count(self, tmp_path: Path) -> None:
        """CSV の行数が正しい."""
        rows = [
            DetectionResultRow(
                image_name=f"img{i}.jpg",
                detection_id=1,
                class_name="cat",
                confidence=0.9,
                x_min=0.0,
                y_min=0.0,
                x_max=100.0,
                y_max=100.0,
                status="",
                iou="",
                gt_class_name="",
            )
            for i in range(5)
        ]
        csv_path = write_detection_results_csv(tmp_path, rows)

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            lines = list(reader)
        # header + 5 data rows
        assert len(lines) == 6

    def test_empty_rows_writes_header_only(self, tmp_path: Path) -> None:
        """空の行リストでもヘッダーのみの CSV が作成される."""
        csv_path = write_detection_results_csv(tmp_path, [])

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            lines = list(reader)
        assert len(lines) == 1  # header only


class TestDetectionResultRowToDict:
    """DetectionResultRow.to_dict のテスト."""

    def test_confidence_formatted(self) -> None:
        """Confidence が 4 桁小数でフォーマットされる."""
        row = DetectionResultRow(
            image_name="test.jpg",
            detection_id=1,
            class_name="cat",
            confidence=0.9512345,
            x_min=10.0,
            y_min=20.0,
            x_max=100.0,
            y_max=200.0,
            status="TP",
            iou=0.85,
            gt_class_name="cat",
        )
        d = row.to_dict()
        assert d["confidence"] == "0.9512"

    def test_empty_confidence_for_fn(self) -> None:
        """FN 行の confidence が空文字."""
        row = DetectionResultRow(
            image_name="test.jpg",
            detection_id=0,
            class_name="",
            confidence="",
            x_min=10.0,
            y_min=20.0,
            x_max=100.0,
            y_max=200.0,
            status="FN",
            iou="",
            gt_class_name="cat",
        )
        d = row.to_dict()
        assert d["confidence"] == ""
        assert d["iou"] == ""
