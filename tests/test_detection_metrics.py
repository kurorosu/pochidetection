"""DetectionMetricsのテスト."""

import pytest
import torch

from pochidetection.interfaces import IDetectionMetrics
from pochidetection.metrics import DetectionMetrics


class TestDetectionMetrics:
    """DetectionMetricsのテスト."""

    @pytest.fixture
    def metrics(self) -> DetectionMetrics:
        """テスト用の評価指標を作成するfixture."""
        return DetectionMetrics()

    def test_implements_interface(self, metrics: DetectionMetrics) -> None:
        """IDetectionMetricsを実装していることを確認."""
        assert isinstance(metrics, IDetectionMetrics)

    def test_update_and_compute(self, metrics: DetectionMetrics) -> None:
        """updateとcomputeが正しく動作することを確認."""
        # 予測: 1つのボックス
        pred_boxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        pred_scores = [torch.tensor([0.9])]
        pred_labels = [torch.tensor([0])]

        # 正解: 1つのボックス (予測と同じ)
        target_boxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        target_labels = [torch.tensor([0])]

        metrics.update(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            target_boxes=target_boxes,
            target_labels=target_labels,
        )

        result = metrics.compute()

        assert "mAP" in result
        assert "mAP_50" in result
        assert "mAP_75" in result
        # 完全一致なのでmAPは1.0
        assert result["mAP"] == 1.0
        assert result["mAP_50"] == 1.0
        assert result["mAP_75"] == 1.0

    def test_reset(self, metrics: DetectionMetrics) -> None:
        """resetが蓄積した状態をリセットすることを確認."""
        # 予測と正解を追加
        pred_boxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        pred_scores = [torch.tensor([0.9])]
        pred_labels = [torch.tensor([0])]
        target_boxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        target_labels = [torch.tensor([0])]

        metrics.update(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            target_boxes=target_boxes,
            target_labels=target_labels,
        )

        # リセット
        metrics.reset()

        # 新しいデータで計算 (間違った予測)
        pred_boxes_wrong = [torch.tensor([[100.0, 100.0, 150.0, 150.0]])]
        pred_scores_wrong = [torch.tensor([0.9])]
        pred_labels_wrong = [torch.tensor([0])]

        metrics.update(
            pred_boxes=pred_boxes_wrong,
            pred_scores=pred_scores_wrong,
            pred_labels=pred_labels_wrong,
            target_boxes=target_boxes,
            target_labels=target_labels,
        )

        result = metrics.compute()

        # 間違った予測なのでmAPは0
        assert result["mAP"] == 0.0

    def test_multiple_updates(self, metrics: DetectionMetrics) -> None:
        """複数回のupdateで結果が蓄積されることを確認."""
        # 1回目: 正しい予測
        pred_boxes_1 = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        pred_scores_1 = [torch.tensor([0.9])]
        pred_labels_1 = [torch.tensor([0])]
        target_boxes_1 = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        target_labels_1 = [torch.tensor([0])]

        metrics.update(
            pred_boxes=pred_boxes_1,
            pred_scores=pred_scores_1,
            pred_labels=pred_labels_1,
            target_boxes=target_boxes_1,
            target_labels=target_labels_1,
        )

        # 2回目: 間違った予測
        pred_boxes_2 = [torch.tensor([[100.0, 100.0, 150.0, 150.0]])]
        pred_scores_2 = [torch.tensor([0.9])]
        pred_labels_2 = [torch.tensor([0])]
        target_boxes_2 = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        target_labels_2 = [torch.tensor([0])]

        metrics.update(
            pred_boxes=pred_boxes_2,
            pred_scores=pred_scores_2,
            pred_labels=pred_labels_2,
            target_boxes=target_boxes_2,
            target_labels=target_labels_2,
        )

        result = metrics.compute()

        # 1/2が正しいので約0.5 (mAP計算の都合で厳密に0.5にならない)
        assert 0.4 < result["mAP"] < 0.6

    def test_custom_iou_thresholds(self) -> None:
        """カスタムIoU閾値を指定できることを確認."""
        metrics = DetectionMetrics(iou_thresholds=[0.5])

        pred_boxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        pred_scores = [torch.tensor([0.9])]
        pred_labels = [torch.tensor([0])]
        target_boxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        target_labels = [torch.tensor([0])]

        metrics.update(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            target_boxes=target_boxes,
            target_labels=target_labels,
        )

        result = metrics.compute()
        assert "mAP" in result

    def test_batch_with_multiple_images(self, metrics: DetectionMetrics) -> None:
        """複数画像のバッチを処理できることを確認."""
        # 2つの画像に対する予測と正解
        pred_boxes = [
            torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            torch.tensor([[20.0, 20.0, 60.0, 60.0], [30.0, 30.0, 70.0, 70.0]]),
        ]
        pred_scores = [
            torch.tensor([0.9]),
            torch.tensor([0.8, 0.7]),
        ]
        pred_labels = [
            torch.tensor([0]),
            torch.tensor([0, 1]),
        ]
        target_boxes = [
            torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            torch.tensor([[20.0, 20.0, 60.0, 60.0], [30.0, 30.0, 70.0, 70.0]]),
        ]
        target_labels = [
            torch.tensor([0]),
            torch.tensor([0, 1]),
        ]

        metrics.update(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            target_boxes=target_boxes,
            target_labels=target_labels,
        )

        result = metrics.compute()

        assert result["mAP"] == 1.0
