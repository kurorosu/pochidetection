"""学習レポートのプロット."""

from pathlib import Path

import plotly.graph_objects as go
from plotly.io import to_html

from pochidetection.interfaces import IReportPlotter, ITrainingCurvePlotter
from pochidetection.visualization.plotters.constants import (
    LEGEND_CONFIG,
    render_side_by_side_html,
)


class TrainingReportPlotter(IReportPlotter):
    """複数のプロッターを横並びで描画.

    Attributes:
        _plotters: プロッターのリスト.
    """

    def __init__(self, *plotters: ITrainingCurvePlotter) -> None:
        """初期化.

        Args:
            *plotters: プロッターのリスト.
        """
        self._plotters = plotters

    def plot(self, output_path: Path) -> None:
        """複数のグラフを横並びで HTML ファイルに出力.

        Args:
            output_path: 出力先パス.
        """
        # 各プロッターからグラフを生成
        html_parts: list[str] = []
        for i, plotter in enumerate(self._plotters):
            fig = go.Figure()
            for trace in plotter.get_traces():
                fig.add_trace(trace)
            fig.update_layout(
                title=plotter.title,
                xaxis_title="Epoch",
                yaxis_title=plotter.y_axis_label,
                legend=LEGEND_CONFIG,
                hovermode="x unified",
                width=600,
                height=600,
            )
            # 最初のグラフのみ plotly.js を含める
            include_js = "cdn" if i == 0 else False
            html_parts.append(
                to_html(fig, full_html=False, include_plotlyjs=include_js)
            )

        # グラフを横並びで HTML に出力
        graphs_html = "\n".join(f"<div>{html}</div>" for html in html_parts)

        html_content = render_side_by_side_html(
            title="Training Report",
            heading="Training Report",
            body=graphs_html,
        )
        output_path.write_text(html_content, encoding="utf-8")
