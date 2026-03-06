"""COCO アノテーションと推論結果から mAP を計算する評価器."""

from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision

from pochidetection.core.detection import Detection
from pochidetection.utils.benchmark import DetectionMetrics
from pochidetection.utils.coco_utils import (
    CocoGroundTruth,
    load_coco_ground_truth,
    xywh_to_xyxy,
)


class MapEvaluator:
    """COCO アノテーションと推論結果から mAP を計算する.

    Attributes:
        _gt: COCO GT データ.
    """

    def __init__(self, annotation_path: Path) -> None:
        """初期化.

        Args:
            annotation_path: COCO フォーマットのアノテーション JSON パス.
        """
        self._gt: CocoGroundTruth = load_coco_ground_truth(annotation_path)

    def evaluate(self, predictions: dict[str, list[Detection]]) -> DetectionMetrics:
        """推論結果と GT から mAP を計算する.

        学習時 (train.py) の mAP 計算との違い:
        - 座標系: 学習時は正規化座標, 本クラスはピクセル座標.
          pred/GT が同一スケールのため mAP 値への影響はない.
        - threshold: 学習時は 0.2 固定, 推論時は CLI 指定値 (デフォルト 0.5).
          閾値が高いほど低スコア検出が除外され, recall が下がりうる.
        - NMS: 学習時は未適用, 推論時は適用済み.
          重複除去後の予測で評価するため, 学習時と値が異なりうる.
        - バッチ更新: 学習時は 1画像ずつ update, 推論時は全画像まとめて update.
          torchmetrics の仕様上, 結果は同一.

        Note:
            本メソッドは GT の全画像を起点に走査するため,
            「GT の画像セット = 推論フォルダの画像セット」という前提を必要としない.
            predictions に含まれない GT 画像は空の予測 (検出 0 件) として扱われ,
            その GT オブジェクトは False Negative にカウントされる.

        Args:
            predictions: ファイル名をキー, 検出結果リストを値とする辞書.

        Returns:
            mAP@0.5 と mAP@0.5:0.95 を含む DetectionMetrics.
        """
        metric = MeanAveragePrecision(iou_type="bbox")

        preds_list: list[dict[str, torch.Tensor]] = []
        targets_list: list[dict[str, torch.Tensor]] = []

        # GT の全画像を走査し, predictions にない画像は空の予測として扱う.
        # これにより, 推論対象外の GT が False Negative として正しくカウントされる.
        for image_id, filenames in self._gt.filenames_by_image_id.items():
            # predictions からファイル名で検出結果を検索
            detections: list[Detection] = []
            for fn in filenames:
                if fn in predictions:
                    detections = predictions[fn]
                    break

            # 予測
            if detections:
                boxes = torch.tensor([d.box for d in detections], dtype=torch.float32)
                scores = torch.tensor(
                    [d.score for d in detections], dtype=torch.float32
                )
                labels = torch.tensor([d.label for d in detections], dtype=torch.int64)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                scores = torch.zeros((0,), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            preds_list.append({"boxes": boxes, "scores": scores, "labels": labels})

            # GT
            gt_anns = self._gt.gt_by_image_id.get(image_id, [])
            if gt_anns:
                gt_boxes = torch.tensor(
                    [xywh_to_xyxy(ann["bbox"]) for ann in gt_anns],
                    dtype=torch.float32,
                )
                gt_labels = torch.tensor(
                    [
                        self._gt.category_id_to_idx[ann["category_id"]]
                        for ann in gt_anns
                    ],
                    dtype=torch.int64,
                )
            else:
                gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
                gt_labels = torch.zeros((0,), dtype=torch.int64)

            targets_list.append({"boxes": gt_boxes, "labels": gt_labels})

        metric.update(preds_list, targets_list)
        result = metric.compute()

        return DetectionMetrics(
            map_50=result["map_50"].item(),
            map_50_95=result["map"].item(),
        )
