"""プロッターのインターフェース."""

from abc import ABC, abstractmethod
from pathlib import Path

import plotly.graph_objects as go


class IPlotter(ABC):
    """プロッターのインターフェース.

    学習曲線などのグラフを描画するクラスはこのインターフェースを実装する.
    """

    @property
    @abstractmethod
    def title(self) -> str:
        """グラフのタイトルを取得.

        Returns:
            グラフのタイトル.
        """
        pass

    @property
    @abstractmethod
    def y_axis_label(self) -> str:
        """Y軸のラベルを取得.

        Returns:
            Y軸のラベル.
        """
        pass

    @abstractmethod
    def get_traces(self) -> list[go.Scatter]:
        """グラフのトレースを取得.

        Returns:
            plotly の Scatter トレースのリスト.
        """
        pass

    def plot(self, output_path: Path) -> None:
        """グラフを HTML ファイルに出力.

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
