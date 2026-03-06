"""損失曲線のプロット."""

import plotly.graph_objects as go

from pochidetection.interfaces import ITrainingCurvePlotter
from pochidetection.utils import TrainingHistory


class LossPlotter(ITrainingCurvePlotter):
    """Train/Val Loss 曲線を plotly で描画.

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
        return "Loss Curve"

    @property
    def y_axis_label(self) -> str:
        """Y軸のラベルを取得.

        Returns:
            Y軸のラベル.
        """
        return "Loss"

    def get_traces(self) -> list[go.Scatter]:
        """グラフのトレースを取得.

        Returns:
            plotly の Scatter トレースのリスト.
        """
        return [
            go.Scatter(
                x=self._history.epochs,
                y=self._history.train_losses,
                mode="lines+markers",
                name="Train Loss",
                line={"color": "#1f77b4"},
            ),
            go.Scatter(
                x=self._history.epochs,
                y=self._history.val_losses,
                mode="lines+markers",
                name="Val Loss",
                line={"color": "#ff7f0e"},
            ),
        ]
