"""混同行列の構築と Plotly ヒートマップによる可視化."""

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import torch
from plotly.io import to_html
from torchvision.ops import box_iou

from pochidetection.scripts.rtdetr.inference.detection import Detection
from pochidetection.utils.coco_utils import load_coco_ground_truth, xywh_to_xyxy

BACKGROUND_LABEL = "Background"


def build_confusion_matrix(
    predictions: dict[str, list[Detection]],
    annotation_path: Path,
    class_names: list[str],
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """推論結果と GT から混同行列を構築する.

    行が GT クラス, 列が予測クラスを表す.
    最終行/列は Background (FN / FP) を表す.

    Args:
        predictions: ファイル名をキー, 検出結果リストを値とする辞書.
        annotation_path: COCO アノテーション JSON パス.
        class_names: クラス名リスト (Background を含まない).
        iou_threshold: マッチング IoU 閾値.

    Returns:
        (num_classes+1) x (num_classes+1) の混同行列テンソル.
        最後の行/列が Background.
    """
    num_classes = len(class_names)
    matrix = torch.zeros(num_classes + 1, num_classes + 1, dtype=torch.int64)
    bg_idx = num_classes  # Background のインデックス

    gt = load_coco_ground_truth(annotation_path)

    # 全画像を走査
    processed_image_ids: set[int] = set()
    for image_name, detections in predictions.items():
        image_id = gt.image_id_by_filename.get(image_name)
        if image_id is not None:
            processed_image_ids.add(image_id)
        gt_anns = gt.gt_by_image_id.get(image_id, []) if image_id is not None else []

        _update_matrix(
            matrix, detections, gt_anns, gt.category_id_to_idx, iou_threshold, bg_idx
        )

    # predictions に含まれない GT 画像の FN を処理
    for image_id, gt_anns in gt.gt_by_image_id.items():
        if image_id in processed_image_ids:
            continue
        _update_matrix(
            matrix, [], gt_anns, gt.category_id_to_idx, iou_threshold, bg_idx
        )

    return matrix


def _update_matrix(
    matrix: torch.Tensor,
    detections: list[Detection],
    gt_anns: list[dict[str, Any]],
    category_id_to_idx: dict[int, int],
    iou_threshold: float,
    bg_idx: int,
) -> None:
    """1 画像分の検出結果で混同行列を更新する.

    Args:
        matrix: 混同行列テンソル (in-place 更新).
        detections: 検出結果リスト.
        gt_anns: GT アノテーションリスト.
        category_id_to_idx: カテゴリ ID → 連続インデックス.
        iou_threshold: マッチング IoU 閾値.
        bg_idx: Background のインデックス.
    """
    num_det = len(detections)
    num_gt = len(gt_anns)

    if num_det == 0 and num_gt == 0:
        return

    # GT なし → 全検出が FP (Background → 予測クラス)
    if num_gt == 0:
        for det in detections:
            matrix[bg_idx, det.label] += 1
        return

    # 検出なし → 全 GT が FN (GT クラス → Background)
    if num_det == 0:
        for ann in gt_anns:
            gt_label = category_id_to_idx.get(ann["category_id"])
            if gt_label is not None:
                matrix[gt_label, bg_idx] += 1
        return

    pred_boxes = torch.tensor([d.box for d in detections], dtype=torch.float32)
    gt_boxes = torch.tensor(
        [xywh_to_xyxy(ann["bbox"]) for ann in gt_anns],
        dtype=torch.float32,
    )

    iou_matrix = box_iou(pred_boxes, gt_boxes)

    matched_gt: set[int] = set()
    matched_det: set[int] = set()

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
            iou_val = iou_matrix[det_idx, gt_idx].item()
            if iou_val >= iou_threshold and iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = gt_idx

        if best_gt_idx is not None:
            gt_label = category_id_to_idx.get(gt_anns[best_gt_idx]["category_id"])
            if gt_label is not None:
                # GT クラス → 予測クラス (TP or クラス間混同)
                matrix[gt_label, det.label] += 1
                matched_gt.add(best_gt_idx)
                matched_det.add(det_idx)

    # 未マッチの検出 → FP (Background → 予測クラス)
    for det_idx in range(num_det):
        if det_idx not in matched_det:
            matrix[bg_idx, detections[det_idx].label] += 1

    # 未マッチの GT → FN (GT クラス → Background)
    for gt_idx in range(num_gt):
        if gt_idx not in matched_gt:
            gt_label = category_id_to_idx.get(gt_anns[gt_idx]["category_id"])
            if gt_label is not None:
                matrix[gt_label, bg_idx] += 1


class ConfusionMatrixPlotter:
    """混同行列を Plotly ヒートマップで可視化.

    Attributes:
        _matrix: 混同行列テンソル.
        _labels: 表示ラベル (クラス名 + "Background").
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        class_names: list[str],
    ) -> None:
        """初期化.

        Args:
            matrix: (num_classes+1) x (num_classes+1) の混同行列.
            class_names: クラス名リスト (Background を含まない).
        """
        self._matrix = matrix
        self._labels = class_names + [BACKGROUND_LABEL]

    def plot(self, output_path: Path) -> None:
        """混同行列を HTML ファイルに出力.

        Args:
            output_path: 出力先パス.
        """
        fig = self._create_figure()
        chart_html = to_html(fig, full_html=False, include_plotlyjs="cdn")
        html_content = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            '<head><meta charset="utf-8"></head>\n'
            "<body>\n"
            '<div style="display:flex;justify-content:center;align-items:center;'
            'min-height:100vh;">\n'
            f"{chart_html}\n"
            "</div>\n"
            "</body>\n"
            "</html>"
        )
        output_path.write_text(html_content, encoding="utf-8")

    def _create_figure(self) -> go.Figure:
        """Plotly Figure を作成.

        Returns:
            plotly Figure オブジェクト.
        """
        z = self._matrix.numpy().tolist()
        labels = self._labels

        # ホバーテキスト: "GT: cat / Pred: dog: 5"
        hover_text = [
            [
                f"GT: {labels[i]}<br>Pred: {labels[j]}<br>Count: {z[i][j]}"
                for j in range(len(labels))
            ]
            for i in range(len(labels))
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                text=hover_text,
                hoverinfo="text",
                texttemplate="%{z}",
                colorscale="Blues",
            )
        )

        size = max(500, 100 * len(labels))
        fig.update_layout(
            title={"text": "Confusion Matrix", "x": 0.5, "xanchor": "center"},
            xaxis_title="Predicted label",
            yaxis_title="True label",
            width=size,
            height=size,
            xaxis={"side": "bottom"},
            yaxis={"autorange": "reversed"},
            margin={"l": 100, "r": 100, "t": 80, "b": 80, "autoexpand": True},
        )

        return fig
