"""推論結果を CSV ファイルとして出力するモジュール."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torchvision.ops import box_iou

from pochidetection.scripts.rtdetr.inference.detection import Detection
from pochidetection.utils.category_utils import (
    build_category_id_to_idx,
    filter_categories,
)
from pochidetection.utils.map_evaluator import MapEvaluator
from pochidetection.visualization import LabelMapper

CSV_COLUMNS = [
    "image_name",
    "detection_id",
    "class_name",
    "confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "status",
    "iou",
    "gt_class_name",
]


@dataclass
class DetectionResultRow:
    """CSV 1行分の検出結果.

    Attributes:
        image_name: 画像ファイル名.
        detection_id: 検出ID (1-indexed). FN の場合は 0.
        class_name: 予測クラス名. FN の場合は空文字.
        confidence: 信頼度. FN の場合は空文字.
        x_min: bbox 左上 x. FN の場合は GT の値.
        y_min: bbox 左上 y. FN の場合は GT の値.
        x_max: bbox 右下 x. FN の場合は GT の値.
        y_max: bbox 右下 y. FN の場合は GT の値.
        status: TP / FP / FN / 空文字 (アノテーションなし).
        iou: GT との IoU. 該当なしの場合は空文字.
        gt_class_name: マッチした GT のクラス名. 該当なしの場合は空文字.
    """

    image_name: str
    detection_id: int | str
    class_name: str
    confidence: float | str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    status: str
    iou: float | str
    gt_class_name: str

    def to_dict(self) -> dict[str, object]:
        """CSV 書き込み用の辞書に変換."""
        return {
            "image_name": self.image_name,
            "detection_id": self.detection_id,
            "class_name": self.class_name,
            "confidence": (
                f"{self.confidence:.4f}" if isinstance(self.confidence, float) else ""
            ),
            "x_min": f"{self.x_min:.1f}",
            "y_min": f"{self.y_min:.1f}",
            "x_max": f"{self.x_max:.1f}",
            "y_max": f"{self.y_max:.1f}",
            "status": self.status,
            "iou": f"{self.iou:.4f}" if isinstance(self.iou, float) else "",
            "gt_class_name": self.gt_class_name,
        }


def _load_ground_truth(
    annotation_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[int, int], list[dict[str, Any]]]:
    """COCO アノテーションから GT を読み込む.

    Args:
        annotation_path: COCO JSON パス.

    Returns:
        (gt_by_filename, category_id_to_idx, categories) のタプル.
        gt_by_filename: ファイル名をキーとした GT アノテーションリスト.
        category_id_to_idx: カテゴリID から連続インデックスへのマッピング.
        categories: フィルタ済みカテゴリリスト.
    """
    with open(annotation_path, encoding="utf-8") as f:
        coco: dict[str, Any] = json.load(f)

    image_id_to_filename: dict[int, str] = {}
    for img in coco["images"]:
        image_id = img["id"]
        file_name = img["file_name"]
        image_id_to_filename[image_id] = file_name
        basename = MapEvaluator._extract_basename(file_name)
        if basename != file_name and basename not in image_id_to_filename.values():
            image_id_to_filename[image_id] = basename

    categories = filter_categories(coco.get("categories", []))
    category_id_to_idx = build_category_id_to_idx(categories)

    gt_by_filename: dict[str, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in category_id_to_idx:
            continue
        image_id = ann["image_id"]
        filename = image_id_to_filename.get(image_id, "")
        if not filename:
            continue
        gt_by_filename.setdefault(filename, []).append(ann)

    return gt_by_filename, category_id_to_idx, categories


def _match_detections(
    detections: list[Detection],
    gt_annotations: list[dict[str, Any]],
    category_id_to_idx: dict[int, int],
    iou_threshold: float,
) -> list[tuple[str, float, int | None]]:
    """検出と GT をマッチングし, TP/FP を判定する.

    Args:
        detections: 検出結果リスト.
        gt_annotations: GT アノテーションリスト.
        category_id_to_idx: カテゴリID から連続インデックスへのマッピング.
        iou_threshold: TP 判定の IoU 閾値.

    Returns:
        各検出に対応する (status, iou, gt_index) のリスト.
        gt_index は マッチした GT のインデックス. FP の場合は None.
    """
    num_det = len(detections)
    num_gt = len(gt_annotations)

    if num_det == 0:
        return []

    if num_gt == 0:
        return [("FP", 0.0, None)] * num_det

    pred_boxes = torch.tensor([d.box for d in detections], dtype=torch.float32)
    gt_boxes = torch.tensor(
        [MapEvaluator._xywh_to_xyxy(ann["bbox"]) for ann in gt_annotations],
        dtype=torch.float32,
    )

    iou_matrix = box_iou(pred_boxes, gt_boxes)

    results: list[tuple[str, float, int | None]] = []
    matched_gt: set[int] = set()

    # 信頼度の高い検出から順にマッチング
    sorted_indices = sorted(
        range(num_det), key=lambda i: detections[i].score, reverse=True
    )

    for det_idx in sorted_indices:
        det = detections[det_idx]
        best_iou = 0.0
        best_gt_idx: int | None = None

        for gt_idx in range(num_gt):
            if gt_idx in matched_gt:
                continue
            gt_label = category_id_to_idx.get(gt_annotations[gt_idx]["category_id"])
            if gt_label is None:
                continue
            if det.label != gt_label:
                continue
            iou_val = iou_matrix[det_idx, gt_idx].item()
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            results.append(("TP", best_iou, best_gt_idx))
            matched_gt.add(best_gt_idx)
        else:
            results.append(("FP", best_iou if best_gt_idx is not None else 0.0, None))

    # 元の順序に戻す
    ordered_results: list[tuple[str, float, int | None]] = [("", 0.0, None)] * num_det
    for orig_idx, result in zip(sorted_indices, results):
        ordered_results[orig_idx] = result

    return ordered_results


def build_detection_results(
    predictions: dict[str, list[Detection]],
    label_mapper: LabelMapper | None,
    annotation_path: Path | None = None,
    iou_threshold: float = 0.5,
) -> list[DetectionResultRow]:
    """推論結果から CSV 出力用の行リストを構築する.

    Args:
        predictions: ファイル名をキー, 検出結果リストを値とする辞書.
        label_mapper: ラベルマッパー.
        annotation_path: COCO アノテーション JSON パス. None でアノテーションなし.
        iou_threshold: TP/FP 判定の IoU 閾値.

    Returns:
        DetectionResultRow のリスト.
    """
    mapper = label_mapper if label_mapper is not None else LabelMapper()

    gt_by_filename: dict[str, list[dict[str, Any]]] = {}
    category_id_to_idx: dict[int, int] = {}
    idx_to_category_name: dict[int, str] = {}
    has_gt = False

    if annotation_path is not None and annotation_path.exists():
        gt_by_filename, category_id_to_idx, categories = _load_ground_truth(
            annotation_path
        )
        idx_to_category_name = {
            idx: cat["name"]
            for cat in categories
            for cid, idx in category_id_to_idx.items()
            if cid == cat["id"]
        }
        has_gt = True

    rows: list[DetectionResultRow] = []

    for image_name in sorted(predictions.keys()):
        detections = predictions[image_name]
        gt_anns = gt_by_filename.get(image_name, []) if has_gt else []

        if has_gt:
            match_results = _match_detections(
                detections, gt_anns, category_id_to_idx, iou_threshold
            )
        else:
            match_results = []

        # 検出結果行
        for det_idx, det in enumerate(detections):
            status: str = ""
            row_iou: float | str = ""
            gt_class: str = ""

            if has_gt and det_idx < len(match_results):
                status, matched_iou, gt_idx = match_results[det_idx]
                row_iou = matched_iou
                if gt_idx is not None:
                    gt_cat_id = gt_anns[gt_idx]["category_id"]
                    gt_label_idx = category_id_to_idx.get(gt_cat_id)
                    if gt_label_idx is not None:
                        gt_class = idx_to_category_name.get(
                            gt_label_idx, str(gt_label_idx)
                        )

            rows.append(
                DetectionResultRow(
                    image_name=image_name,
                    detection_id=det_idx + 1,
                    class_name=mapper.get_label(det.label),
                    confidence=det.score,
                    x_min=det.box[0],
                    y_min=det.box[1],
                    x_max=det.box[2],
                    y_max=det.box[3],
                    status=status,
                    iou=row_iou,
                    gt_class_name=gt_class,
                )
            )

        # FN 行 (未検出の GT)
        if has_gt:
            matched_gt_indices = {
                r[2] for r in match_results if r[0] == "TP" and r[2] is not None
            }
            for gt_idx, ann in enumerate(gt_anns):
                if gt_idx in matched_gt_indices:
                    continue
                gt_cat_id = ann["category_id"]
                gt_label_idx = category_id_to_idx.get(gt_cat_id)
                if gt_label_idx is None:
                    continue
                gt_class = idx_to_category_name.get(gt_label_idx, str(gt_label_idx))
                gt_box = MapEvaluator._xywh_to_xyxy(ann["bbox"])
                rows.append(
                    DetectionResultRow(
                        image_name=image_name,
                        detection_id=0,
                        class_name="",
                        confidence="",
                        x_min=gt_box[0],
                        y_min=gt_box[1],
                        x_max=gt_box[2],
                        y_max=gt_box[3],
                        status="FN",
                        iou="",
                        gt_class_name=gt_class,
                    )
                )

    return rows


def write_detection_results_csv(
    output_dir: Path,
    rows: list[DetectionResultRow],
) -> Path:
    """検出結果を CSV ファイルに書き出す.

    Args:
        output_dir: 出力ディレクトリ.
        rows: 検出結果行のリスト.

    Returns:
        出力した CSV ファイルのパス.
    """
    csv_path = output_dir / "detection_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    return csv_path
