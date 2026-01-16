"""物体検出用評価指標.

mAP (mean Average Precision) などの評価指標を計算する.
"""

import torch
from torchmetrics.detection import MeanAveragePrecision

from pochidetection.interfaces.metrics import IDetectionMetrics


class DetectionMetrics(IDetectionMetrics):
    """物体検出用評価指標.

    torchmetricsのMeanAveragePrecisionをラップし,
    IDetectionMetricsインターフェースを実装する.

    Attributes:
        _map_metric: torchmetricsのMeanAveragePrecisionインスタンス.
        _iou_thresholds: IoU閾値のリスト.
    """

    def __init__(
        self,
        iou_thresholds: list[float] | None = None,
        box_format: str = "xyxy",
    ) -> None:
        """DetectionMetricsを初期化.

        Args:
            iou_thresholds: mAP計算に使用するIoU閾値のリスト.
                Noneの場合はCOCO標準 [0.5, 0.55, ..., 0.95] を使用.
            box_format: ボックス形式. "xyxy", "xywh", "cxcywh" のいずれか.
        """
        self._iou_thresholds = iou_thresholds
        self._box_format = box_format
        self._map_metric = MeanAveragePrecision(
            iou_thresholds=iou_thresholds,
            box_format=box_format,
        )

    def update(
        self,
        pred_boxes: list[torch.Tensor],
        pred_scores: list[torch.Tensor],
        pred_labels: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
        target_labels: list[torch.Tensor],
    ) -> None:
        """バッチ結果を蓄積.

        Args:
            pred_boxes: 予測ボックスのリスト. 各要素は (N, 4) の形状.
            pred_scores: 予測スコアのリスト. 各要素は (N,) の形状.
            pred_labels: 予測ラベルのリスト. 各要素は (N,) の形状.
            target_boxes: 正解ボックスのリスト. 各要素は (M, 4) の形状.
            target_labels: 正解ラベルのリスト. 各要素は (M,) の形状.
        """
        # torchmetricsの形式に変換
        preds = [
            {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
            for boxes, scores, labels in zip(
                pred_boxes, pred_scores, pred_labels, strict=True
            )
        ]
        targets = [
            {
                "boxes": boxes,
                "labels": labels,
            }
            for boxes, labels in zip(target_boxes, target_labels, strict=True)
        ]

        self._map_metric.update(preds, targets)

    def compute(self) -> dict[str, float]:
        """蓄積した結果から指標を計算.

        Returns:
            指標名と値の辞書:
            - mAP: COCO標準の平均mAP (IoU 0.5:0.95)
            - mAP_50: IoU 0.5でのmAP
            - mAP_75: IoU 0.75でのmAP
        """
        result = self._map_metric.compute()

        return {
            "mAP": result["map"].item(),
            "mAP_50": result["map_50"].item(),
            "mAP_75": result["map_75"].item(),
        }

    def reset(self) -> None:
        """蓄積した状態をリセット."""
        self._map_metric.reset()
