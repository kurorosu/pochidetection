"""PR曲線のプロット."""

from pathlib import Path

import plotly.graph_objects as go
import torch
from plotly.io import to_html

# 凡例設定（グラフ右上外）
LEGEND_CONFIG = {
    "yanchor": "top",
    "y": 1,
    "xanchor": "left",
    "x": 1.02,
}


class PRCurvePlotter:
    """PR曲線をplotlyで描画.

    torchmetrics.MeanAveragePrecision(extended_summary=True) の結果から
    Precision-Recall 曲線を描画する.

    Attributes:
        _precision: Precision テンソル (TxRxKxAxM).
        _recall_thresholds: Recall 閾値のリスト.
        _class_names: クラス名のリスト.
    """

    # COCO標準のRecall閾値 (101点)
    RECALL_THRESHOLDS = torch.linspace(0.0, 1.0, 101)

    def __init__(
        self,
        precision: torch.Tensor,
        class_names: list[str] | None = None,
        iou_threshold_idx: int = 0,
    ) -> None:
        """初期化.

        Args:
            precision: Precision テンソル (TxRxKxAxM).
                T=IoU閾値数, R=Recall閾値数, K=クラス数, A=領域数, M=最大検出数.
            class_names: クラス名のリスト. None の場合は "Class 0", "Class 1", ... を使用.
            iou_threshold_idx: 使用するIoU閾値のインデックス (デフォルト: 0 = IoU=0.50).
        """
        self._precision = precision
        self._iou_threshold_idx = iou_threshold_idx
        self._num_classes = precision.shape[2]

        if class_names is None:
            self._class_names = [f"Class {i}" for i in range(self._num_classes)]
        else:
            self._class_names = class_names

    def plot(self, output_path: Path) -> None:
        """PR曲線をHTMLファイルに出力.

        Args:
            output_path: 出力先パス.
        """
        # 全クラス平均のPR曲線
        all_classes_fig = self._create_all_classes_figure()

        # クラス別のPR曲線
        per_class_fig = self._create_per_class_figure()

        # 2つのグラフを横並びでHTMLに出力
        all_classes_html = to_html(
            all_classes_fig, full_html=False, include_plotlyjs="cdn"
        )
        per_class_html = to_html(per_class_fig, full_html=False, include_plotlyjs=False)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>PR Curve</title>
    <style>
        .container {{
            display: flex;
            justify-content: center;
            gap: 20px;
        }}
    </style>
</head>
<body>
    <h1 style="text-align: center;">Precision-Recall Curve</h1>
    <div class="container">
        <div>{all_classes_html}</div>
        <div>{per_class_html}</div>
    </div>
</body>
</html>
"""
        output_path.write_text(html_content, encoding="utf-8")

    def _create_all_classes_figure(self) -> go.Figure:
        """全クラス平均のPR曲線を作成.

        Returns:
            plotly Figure オブジェクト.
        """
        fig = go.Figure()

        # precision: (T, R, K, A, M) -> IoU=iou_threshold_idx, area=all(0), maxDets=100(2)
        # 全クラスの平均を計算
        precision_slice = self._precision[self._iou_threshold_idx, :, :, 0, 2]
        # 無効値(-1)を除外して平均
        valid_mask = precision_slice >= 0
        precision_mean = torch.where(
            valid_mask, precision_slice, torch.full_like(precision_slice, float("nan"))
        ).nanmean(dim=1)

        recall_values = self.RECALL_THRESHOLDS.numpy()
        precision_values = precision_mean.numpy()

        fig.add_trace(
            go.Scatter(
                x=recall_values,
                y=precision_values,
                mode="lines",
                name="All Classes (mean)",
                line={"width": 2},
            )
        )

        fig.update_layout(
            title="PR Curve (All Classes Average)",
            xaxis_title="Recall",
            yaxis_title="Precision",
            legend=LEGEND_CONFIG,
            hovermode="x unified",
            width=600,
            height=600,
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
        )

        return fig

    def _create_per_class_figure(self) -> go.Figure:
        """クラス別のPR曲線を作成.

        Note:
            torchmetrics は GT が 0 件のクラスの precision を -1 (評価不能) にする.
            例: categories が ["cat", "dog", "bird"] で val データに bird の GT が
            存在しない場合, bird の全 recall ポイントが -1 になる.
            -1 をそのまま描画すると不正な値がプロットされるため, NaN に置換する.

        Returns:
            plotly Figure オブジェクト.
        """
        fig = go.Figure()

        recall_values = self.RECALL_THRESHOLDS.numpy()

        for class_idx in range(self._num_classes):
            # precision: (T, R, K, A, M) -> IoU=iou_threshold_idx, class=class_idx,
            # area=all(0), maxDets=100(2)
            precision_slice = self._precision[
                self._iou_threshold_idx, :, class_idx, 0, 2
            ]

            # 全ポイントが無効値(-1)のクラスはスキップ
            if (precision_slice < 0).all():
                continue

            # 無効値(-1)を NaN に置換して描画から除外
            precision_values = torch.where(
                precision_slice >= 0,
                precision_slice,
                torch.full_like(precision_slice, float("nan")),
            ).numpy()

            fig.add_trace(
                go.Scatter(
                    x=recall_values,
                    y=precision_values,
                    mode="lines",
                    name=self._class_names[class_idx],
                    line={"width": 2},
                )
            )

        fig.update_layout(
            title="PR Curve (Per Class)",
            xaxis_title="Recall",
            yaxis_title="Precision",
            legend=LEGEND_CONFIG,
            hovermode="x unified",
            width=600,
            height=600,
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
        )

        return fig
