"""mAP曲線のプロット."""

from pathlib import Path

import plotly.graph_objects as go

from pochidetection.interfaces import IPlotter
from pochidetection.utils import TrainingHistory


class MetricsPlotter(IPlotter):
    """mAP曲線をplotlyで描画.

    Attributes:
        _history: 学習履歴.
    """

    def __init__(self, history: TrainingHistory) -> None:
        """初期化.

        Args:
            history: 学習履歴.
        """
        self._history = history

    @property
    def title(self) -> str:
        """グラフのタイトルを取得.

        Returns:
            グラフのタイトル.
        """
        return "mAP Curve"

    @property
    def y_axis_label(self) -> str:
        """Y軸のラベルを取得.

        Returns:
            Y軸のラベル.
        """
        return "mAP"

    def get_traces(self) -> list[go.Scatter]:
        """グラフのトレースを取得.

        Returns:
            plotly の Scatter トレースのリスト.
        """
        return [
            go.Scatter(
                x=self._history.epochs,
                y=self._history.mAPs,
                mode="lines+markers",
                name="mAP",
                line={"color": "#2ca02c"},
            ),
            go.Scatter(
                x=self._history.epochs,
                y=self._history.mAP_50s,
                mode="lines+markers",
                name="mAP@50",
                line={"color": "#9467bd"},
            ),
            go.Scatter(
                x=self._history.epochs,
                y=self._history.mAP_75s,
                mode="lines+markers",
                name="mAP@75",
                line={"color": "#8c564b"},
            ),
        ]

    def plot(self, output_path: Path) -> None:
        """mAP曲線をHTMLファイルに出力.

        Args:
            output_path: 出力先パス.
        """
        fig = go.Figure()

        for trace in self.get_traces():
            fig.add_trace(trace)

        fig.update_layout(
            title=self.title,
            xaxis_title="Epoch",
            yaxis_title=self.y_axis_label,
            legend={"x": 1, "y": 1, "xanchor": "right", "bgcolor": "rgba(0,0,0,0)"},
            hovermode="x unified",
            width=600,
            height=600,
        )

        fig.write_html(output_path)
