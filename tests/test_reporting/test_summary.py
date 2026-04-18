"""検出サマリースキーマ・ビルダー・ライターのテスト."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pochidetection.core.detection import Detection
from pochidetection.pipelines.builder import _log_detection_summary
from pochidetection.reporting.summary import (
    DETECTION_SUMMARY_SCHEMA_VERSION,
    ClassCount,
    DetectionSummary,
    build_detection_summary,
    write_detection_summary,
)
from pochidetection.visualization import LabelMapper

# ---------- スキーマテスト ----------


class TestClassCount:
    """ClassCount のフィールド検証テスト."""

    def test_valid_construction(self) -> None:
        """正常な値で構築できることを確認."""
        cc = ClassCount(
            label=0, name="cat", count=10, avg_score=0.85, images_with_detections=5
        )
        assert cc.label == 0
        assert cc.name == "cat"
        assert cc.count == 10
        assert cc.avg_score == 0.85
        assert cc.images_with_detections == 5

    def test_extra_field_raises_validation_error(self) -> None:
        """extra='forbid' で未知フィールドが拒否されることを確認."""
        with pytest.raises(ValidationError):
            ClassCount(
                label=0,
                name="cat",
                count=10,
                avg_score=0.85,
                images_with_detections=5,
                unknown="bad",  # type: ignore[call-arg]
            )


class TestDetectionSummary:
    """DetectionSummary のラウンドトリップテスト."""

    def test_model_dump_json_round_trip(self) -> None:
        """model_dump_json() -> JSON パース -> model_validate で復元できることを確認."""
        original = DetectionSummary(
            total_images=10,
            total_detections=25,
            per_class=[
                ClassCount(
                    label=0,
                    name="cat",
                    count=15,
                    avg_score=0.9,
                    images_with_detections=7,
                ),
                ClassCount(
                    label=1,
                    name="dog",
                    count=10,
                    avg_score=0.8,
                    images_with_detections=5,
                ),
            ],
            images_without_detections=2,
        )

        json_str = original.model_dump_json(indent=2)
        parsed = json.loads(json_str)
        restored = DetectionSummary.model_validate(parsed)

        assert restored == original
        assert restored.schema_version == DETECTION_SUMMARY_SCHEMA_VERSION

    def test_extra_field_raises_validation_error(self) -> None:
        """extra='forbid' で未知フィールドが拒否されることを確認."""
        with pytest.raises(ValidationError):
            DetectionSummary(
                total_images=1,
                total_detections=0,
                per_class=[],
                images_without_detections=1,
                unknown="bad",  # type: ignore[call-arg]
            )


# ---------- ビルダーテスト ----------


class TestBuildDetectionSummary:
    """build_detection_summary のテスト."""

    def test_multiple_classes(self) -> None:
        """複数クラスの検出結果を正しく集計できることを確認."""
        predictions: dict[str, list[Detection]] = {
            "img1.jpg": [
                Detection(box=[0, 0, 10, 10], score=0.9, label=0),
                Detection(box=[20, 20, 30, 30], score=0.8, label=1),
            ],
            "img2.jpg": [
                Detection(box=[0, 0, 10, 10], score=0.7, label=0),
            ],
            "img3.jpg": [],
        }
        mapper = LabelMapper(["cat", "dog"])

        summary = build_detection_summary(predictions, mapper)

        assert summary.total_images == 3
        assert summary.total_detections == 3
        assert summary.images_without_detections == 1
        assert len(summary.per_class) == 2

        cat = summary.per_class[0]
        assert cat.label == 0
        assert cat.name == "cat"
        assert cat.count == 2
        assert cat.avg_score == 0.8
        assert cat.images_with_detections == 2

        dog = summary.per_class[1]
        assert dog.label == 1
        assert dog.name == "dog"
        assert dog.count == 1
        assert dog.avg_score == 0.8
        assert dog.images_with_detections == 1

    def test_no_detections(self) -> None:
        """全画像で検出0件の場合を正しく集計できることを確認."""
        predictions: dict[str, list[Detection]] = {
            "img1.jpg": [],
            "img2.jpg": [],
        }

        summary = build_detection_summary(predictions, None)

        assert summary.total_images == 2
        assert summary.total_detections == 0
        assert summary.images_without_detections == 2
        assert summary.per_class == []

    def test_empty_predictions(self) -> None:
        """空の predictions を正しく処理できることを確認."""
        summary = build_detection_summary({}, None)

        assert summary.total_images == 0
        assert summary.total_detections == 0
        assert summary.images_without_detections == 0
        assert summary.per_class == []

    def test_label_mapper_none_fallback(self) -> None:
        """LabelMapper が None の場合にクラスIDの整数文字列でフォールバックすることを確認."""
        predictions: dict[str, list[Detection]] = {
            "img1.jpg": [
                Detection(box=[0, 0, 10, 10], score=0.9, label=2),
            ],
        }

        summary = build_detection_summary(predictions, None)

        assert summary.per_class[0].name == "2"

    def test_per_class_sorted_by_label(self) -> None:
        """per_class が label の昇順でソートされていることを確認."""
        predictions: dict[str, list[Detection]] = {
            "img1.jpg": [
                Detection(box=[0, 0, 10, 10], score=0.5, label=2),
                Detection(box=[0, 0, 10, 10], score=0.5, label=0),
                Detection(box=[0, 0, 10, 10], score=0.5, label=1),
            ],
        }

        summary = build_detection_summary(predictions, None)

        labels = [cc.label for cc in summary.per_class]
        assert labels == [0, 1, 2]


# ---------- ライターテスト ----------


class TestWriteDetectionSummary:
    """write_detection_summary のテスト."""

    def test_writes_json_and_round_trips(self, tmp_path: Path) -> None:
        """JSON ファイルを書き出し, 読み戻しで復元できることを確認."""
        summary = build_detection_summary(
            {
                "img1.jpg": [
                    Detection(box=[0, 0, 10, 10], score=0.9, label=0),
                ],
            },
            LabelMapper(["cat"]),
        )

        output_path = write_detection_summary(tmp_path, summary)

        assert output_path.exists()
        assert output_path.name == "detection_summary.json"

        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        restored = DetectionSummary.model_validate(loaded)
        assert restored == summary


# ---------- _log_detection_summary テスト ----------


class TestLogDetectionSummary:
    """_log_detection_summary のテスト."""

    def test_runs_without_error(self) -> None:
        """正常な DetectionSummary でエラーなく実行されることを確認."""
        summary = DetectionSummary(
            total_images=10,
            total_detections=25,
            per_class=[
                ClassCount(
                    label=0,
                    name="cat",
                    count=15,
                    avg_score=0.9,
                    images_with_detections=7,
                ),
            ],
            images_without_detections=3,
        )
        _log_detection_summary(summary)

    def test_empty_summary_runs_without_error(self) -> None:
        """空の DetectionSummary でもエラーなく実行されることを確認."""
        summary = DetectionSummary(
            total_images=0,
            total_detections=0,
            per_class=[],
            images_without_detections=0,
        )
        _log_detection_summary(summary)
