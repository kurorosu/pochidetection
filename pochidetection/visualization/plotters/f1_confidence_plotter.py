"""F1-Confidence 曲線のプロット."""

from pathlib import Path

import plotly.graph_objects as go
import torch
from plotly.io import to_html

from pochidetection.interfaces import IReportPlotter
from pochidetection.visualization.plotters.constants import (
    LEGEND_CONFIG,
    render_side_by_side_html,
)


class F1ConfidencePlotter(IReportPlotter):
    """F1-Confidence 曲線を plotly で描画.

    torchmetrics.MeanAveragePrecision(extended_summary=True) の結果から
    信頼度閾値ごとの F1 スコア変化を可視化する.

    Attributes:
        _precision: Precision テンソル (TxRxKxAxM).
        _scores: Scores テンソル (TxRxKxAxM).
        _class_names: クラス名のリスト.
        _iou_threshold_idx: 使用する IoU 閾値のインデックス.
    """

    RECALL_THRESHOLDS = torch.linspace(0.0, 1.0, 101)

    def __init__(
        self,
        precision: torch.Tensor,
        scores: torch.Tensor,
        class_names: list[str] | None = None,
        iou_threshold_idx: int = 0,
    ) -> None:
        """初期化.

        Args:
            precision: Precision テンソル (TxRxKxAxM).
                T=IoU閾値数, R=Recall閾値数, K=クラス数, A=領域数, M=最大検出数.
            scores: Scores テンソル (TxRxKxAxM). 各ポイントの Confidence 値.
            class_names: クラス名のリスト. None の場合は "Class 0", "Class 1", ... を使用.
            iou_threshold_idx: 使用するIoU閾値のインデックス (デフォルト: 0 = IoU=0.50).
        """
        self._precision = precision
        self._scores = scores
        self._iou_threshold_idx = iou_threshold_idx
        self._num_classes = precision.shape[2]

        if class_names is None:
            self._class_names = [f"Class {i}" for i in range(self._num_classes)]
        else:
            self._class_names = class_names

    def plot(self, output_path: Path) -> None:
        """F1-Confidence 曲線を HTML ファイルに出力.

        Args:
            output_path: 出力先パス.
        """
        all_classes_fig = self._create_all_classes_figure()
        per_class_fig = self._create_per_class_figure()

        all_classes_html = to_html(
            all_classes_fig, full_html=False, include_plotlyjs="cdn"
        )
        per_class_html = to_html(per_class_fig, full_html=False, include_plotlyjs=False)

        body = f"<div>{all_classes_html}</div>\n        <div>{per_class_html}</div>"
        html_content = render_side_by_side_html(
            title="F1-Confidence Curve",
            heading="F1-Confidence Curve",
            body=body,
        )
        output_path.write_text(html_content, encoding="utf-8")

    def _compute_f1(
        self, precision_slice: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precision スライスから F1 と Recall を計算.

        Args:
            precision_slice: Precision 値 (R,) または (R, K).

        Returns:
            (f1, recall) のタプル. 同じ形状.
        """
        recall = self.RECALL_THRESHOLDS
        if precision_slice.dim() == 2:
            recall = recall[:, None]
        eps = 1e-16
        f1 = 2 * precision_slice * recall / (precision_slice + recall + eps)
        return f1, self.RECALL_THRESHOLDS

    def _create_all_classes_figure(self) -> go.Figure:
        """全クラス平均の F1-Confidence 曲線を作成.

        Returns:
            plotly Figure オブジェクト.
        """
        fig = go.Figure()

        # (R, K) スライス: IoU=0.5, area=all, maxDets=100
        precision_slice = self._precision[self._iou_threshold_idx, :, :, 0, 2]
        scores_slice = self._scores[self._iou_threshold_idx, :, :, 0, 2]

        # 無効値 (-1) を NaN に置換
        valid_mask = precision_slice >= 0
        precision_clean = torch.where(
            valid_mask, precision_slice, torch.full_like(precision_slice, float("nan"))
        )
        scores_clean = torch.where(
            valid_mask, scores_slice, torch.full_like(scores_slice, float("nan"))
        )

        # F1 計算: (R, K)
        f1, _ = self._compute_f1(precision_clean)

        # クラス平均
        f1_mean = f1.nanmean(dim=1)
        scores_mean = scores_clean.nanmean(dim=1)

        # NaN を除いた有効なポイントのみでプロット
        valid = ~(torch.isnan(f1_mean) | torch.isnan(scores_mean))
        conf_values = scores_mean[valid].numpy()
        f1_values = f1_mean[valid].numpy()

        fig.add_trace(
            go.Scatter(
                x=conf_values,
                y=f1_values,
                mode="lines",
                name="All Classes (mean)",
                line={"width": 2},
            )
        )

        # 最大 F1 のマーカー
        if len(f1_values) > 0:
            best_idx = f1_values.argmax()
            best_conf = conf_values[best_idx]
            best_f1 = f1_values[best_idx]
            fig.add_trace(
                go.Scatter(
                    x=[best_conf],
                    y=[best_f1],
                    mode="markers+text",
                    name=f"Best F1={best_f1:.3f} @ {best_conf:.3f}",
                    marker={"size": 10, "symbol": "star"},
                    text=[f"F1={best_f1:.3f}"],
                    textposition="top center",
                    showlegend=True,
                )
            )

        fig.update_layout(
            title="F1-Confidence Curve (All Classes Average)",
            xaxis_title="Confidence",
            yaxis_title="F1 Score",
            legend=LEGEND_CONFIG,
            hovermode="x unified",
            width=600,
            height=600,
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
        )

        return fig

    def _create_per_class_figure(self) -> go.Figure:
        """クラス別の F1-Confidence 曲線を作成.

        Returns:
            plotly Figure オブジェクト.
        """
        fig = go.Figure()

        for class_idx in range(self._num_classes):
            # (R,) スライス
            precision_slice = self._precision[
                self._iou_threshold_idx, :, class_idx, 0, 2
            ]
            scores_slice = self._scores[self._iou_threshold_idx, :, class_idx, 0, 2]

            # 全ポイントが無効値 (-1) のクラスはスキップ
            if (precision_slice < 0).all():
                continue

            # 無効値を NaN に置換
            valid_mask = precision_slice >= 0
            precision_clean = torch.where(
                precision_slice >= 0,
                precision_slice,
                torch.full_like(precision_slice, float("nan")),
            )
            scores_clean = torch.where(
                valid_mask,
                scores_slice,
                torch.full_like(scores_slice, float("nan")),
            )

            # F1 計算: (R,)
            f1, _ = self._compute_f1(precision_clean)

            # NaN を除いた有効なポイント
            valid = ~(torch.isnan(f1) | torch.isnan(scores_clean))
            conf_values = scores_clean[valid].numpy()
            f1_values = f1[valid].numpy()

            fig.add_trace(
                go.Scatter(
                    x=conf_values,
                    y=f1_values,
                    mode="lines",
                    name=self._class_names[class_idx],
                    line={"width": 2},
                )
            )

        fig.update_layout(
            title="F1-Confidence Curve (Per Class)",
            xaxis_title="Confidence",
            yaxis_title="F1 Score",
            legend=LEGEND_CONFIG,
            hovermode="x unified",
            width=600,
            height=600,
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
        )

        return fig
