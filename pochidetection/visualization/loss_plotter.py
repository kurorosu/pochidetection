"""損失曲線のプロット."""

from pathlib import Path

import plotly.graph_objects as go

from pochidetection.utils import TrainingHistory


class LossPlotter:
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

    def plot(self, output_path: Path) -> None:
        """損失曲線を HTML ファイルに出力.

        Args:
            output_path: 出力先パス.
        """
        fig = go.Figure()

        # Train Loss
        fig.add_trace(
            go.Scatter(
                x=self._history.epochs,
                y=self._history.train_losses,
                mode="lines+markers",
                name="Train Loss",
                line={"color": "#1f77b4"},
            )
        )

        # Val Loss
        fig.add_trace(
            go.Scatter(
                x=self._history.epochs,
                y=self._history.val_losses,
                mode="lines+markers",
                name="Val Loss",
                line={"color": "#ff7f0e"},
            )
        )

        fig.update_layout(
            title="Loss Curve",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            legend={"x": 1, "y": 1, "xanchor": "right"},
            hovermode="x unified",
            width=600,
            height=600,
        )

        fig.write_html(output_path)
