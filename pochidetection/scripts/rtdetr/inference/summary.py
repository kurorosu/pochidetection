"""検出サマリースキーマ・構築・出力."""

from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from pochidetection.core.detection import Detection
from pochidetection.visualization import LabelMapper

DETECTION_SUMMARY_FILENAME = "detection_summary.json"
DETECTION_SUMMARY_SCHEMA_VERSION = "1.0.0"


class ClassCount(BaseModel):
    """クラス毎の検出集計."""

    model_config = ConfigDict(extra="forbid")

    label: int
    name: str
    count: int
    avg_score: float
    images_with_detections: int


class DetectionSummary(BaseModel):
    """detection_summary.json のルートスキーマ."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = DETECTION_SUMMARY_SCHEMA_VERSION
    total_images: int
    total_detections: int
    per_class: list[ClassCount]
    images_without_detections: int


def build_detection_summary(
    all_predictions: dict[str, list[Detection]],
    label_mapper: LabelMapper | None,
) -> DetectionSummary:
    """全画像の検出結果からクラス毎の検出サマリーを構築する.

    Args:
        all_predictions: ファイル名をキー, 検出結果リストを値とする辞書.
        label_mapper: クラスIDからクラス名への変換器. None の場合は整数文字列.

    Returns:
        構築した DetectionSummary.
    """
    total_images = len(all_predictions)
    total_detections = 0
    images_without_detections = 0

    counts: dict[int, int] = defaultdict(int)
    scores: dict[int, list[float]] = defaultdict(list)
    image_sets: dict[int, set[str]] = defaultdict(set)

    for filename, detections in all_predictions.items():
        if not detections:
            images_without_detections += 1
            continue

        for det in detections:
            total_detections += 1
            counts[det.label] += 1
            scores[det.label].append(det.score)
            image_sets[det.label].add(filename)

    per_class = []
    for label in sorted(counts.keys()):
        name = label_mapper.get_label(label) if label_mapper else str(label)
        label_scores = scores[label]
        avg_score = sum(label_scores) / len(label_scores)
        per_class.append(
            ClassCount(
                label=label,
                name=name,
                count=counts[label],
                avg_score=round(avg_score, 4),
                images_with_detections=len(image_sets[label]),
            )
        )

    return DetectionSummary(
        total_images=total_images,
        total_detections=total_detections,
        per_class=per_class,
        images_without_detections=images_without_detections,
    )


def write_detection_summary(
    output_dir: Path,
    summary: DetectionSummary,
    filename: str = DETECTION_SUMMARY_FILENAME,
) -> Path:
    """検出サマリーを JSON ファイルに書き出す.

    Args:
        output_dir: 出力ディレクトリ.
        summary: 検出サマリー.
        filename: 出力ファイル名.

    Returns:
        書き出したファイルのパス.
    """
    output_path = output_dir / filename
    output_path.write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return output_path
