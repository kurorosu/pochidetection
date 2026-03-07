"""学習ループの共通ロジック.

RT-DETR と SSDLite で共有されるエポックループ, Early Stopping,
レポート出力のロジックを提供する.
"""

from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from pochidetection.interfaces.model import IDetectionModel
from pochidetection.logging import LoggerManager
from pochidetection.utils import (
    EarlyStopping,
    TrainingHistory,
    WorkspaceManager,
)
from pochidetection.visualization import (
    LossPlotter,
    MetricsPlotter,
    PRCurvePlotter,
    TrainingReportPlotter,
)


class TrainingContext(NamedTuple):
    """学習コンテキスト (アーキテクチャ共通)."""

    model: IDetectionModel
    train_loader: DataLoader  # type: ignore[type-arg]
    val_loader: DataLoader  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    map_metric: MeanAveragePrecision
    workspace: Path
    workspace_manager: WorkspaceManager
    device: str
    epochs: int
    train_score_threshold: float


class ModelSaver(Protocol):
    """モデル保存のプロトコル."""

    def __call__(self, ctx: TrainingContext, save_dir: Path) -> None:
        """モデルを指定ディレクトリに保存する.

        Args:
            ctx: 学習コンテキスト.
            save_dir: 保存先ディレクトリ.
        """
        ...


class Validator(Protocol):
    """検証ループのプロトコル."""

    def __call__(
        self, ctx: TrainingContext, logger: Any
    ) -> tuple[float, dict[str, Any]]:
        """検証を実行して損失と mAP 結果を返す.

        Args:
            ctx: 学習コンテキスト.
            logger: ロガー.

        Returns:
            (平均検証損失, mAP 計算結果辞書) のタプル.
        """
        ...


def run_training_loop(
    config: dict[str, Any],
    ctx: TrainingContext,
    validate: Validator,
    save_model: ModelSaver,
) -> None:
    """共通学習ループ.

    Args:
        config: 設定辞書.
        ctx: 学習コンテキスト.
        validate: 検証関数.
        save_model: モデル保存関数.
    """
    logger = LoggerManager().get_logger(__name__)

    best_map = 0.0
    history = TrainingHistory()
    map_result: dict[str, Any] = {}

    early_stopping = build_early_stopping(config)
    if early_stopping is not None:
        logger.info(
            f"Early Stopping: patience={early_stopping.patience}, "
            f"metric={config['early_stopping_metric']}, "
            f"min_delta={config['early_stopping_min_delta']}"
        )

    for epoch in range(ctx.epochs):
        avg_loss, lr = train_one_epoch(ctx)
        avg_val_loss, map_result = validate(ctx, logger)

        mAP = map_result["map"].item()
        mAP_50 = map_result["map_50"].item()
        mAP_75 = map_result["map_75"].item()

        logger.info(
            f"Epoch {epoch + 1}/{ctx.epochs} - "
            f"Train Loss: {avg_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"mAP: {mAP:.4f}, "
            f"mAP@50: {mAP_50:.4f}, "
            f"mAP@75: {mAP_75:.4f}, "
            f"LR: {lr:.2e}"
        )

        history.add(
            epoch=epoch + 1,
            train_loss=avg_loss,
            val_loss=avg_val_loss,
            mAP=mAP,
            mAP_50=mAP_50,
            mAP_75=mAP_75,
            lr=lr,
        )

        if ctx.scheduler is not None:
            if isinstance(ctx.scheduler, ReduceLROnPlateau):
                ctx.scheduler.step(avg_val_loss)
            else:
                ctx.scheduler.step()

        if early_stopping is not None:
            metric = config["early_stopping_metric"]
            value = mAP if metric == "mAP" else avg_val_loss
            should_stop = early_stopping.step(value, epoch + 1)

            if early_stopping.counter == 0:
                _save_best(ctx, save_model, metric, value, logger)

            if should_stop:
                logger.info(
                    f"Early Stopping: {early_stopping.patience} エポック連続で "
                    f"{metric} が改善しなかったため学習を終了します "
                    f"(best epoch: {early_stopping.best_epoch}, "
                    f"best {metric}: {early_stopping.best_value:.4f})"
                )
                break
        else:
            if mAP > best_map:
                best_map = mAP
                _save_best(ctx, save_model, "mAP", mAP, logger)

    save_results(ctx, history, map_result, save_model, logger)


def build_early_stopping(config: dict[str, Any]) -> EarlyStopping | None:
    """設定から EarlyStopping を構築.

    Args:
        config: 設定辞書.

    Returns:
        EarlyStopping インスタンス, 無効時は None.
    """
    patience = config.get("early_stopping_patience")
    if patience is None:
        return None

    metric = config["early_stopping_metric"]
    mode: Literal["min", "max"] = "max" if metric == "mAP" else "min"
    min_delta = config["early_stopping_min_delta"]

    return EarlyStopping(patience=patience, min_delta=min_delta, mode=mode)


def train_one_epoch(ctx: TrainingContext) -> tuple[float, float]:
    """1エポックの学習.

    Args:
        ctx: 学習コンテキスト.

    Returns:
        (平均学習損失, 現在の学習率) のタプル.
    """
    ctx.model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(ctx.train_loader):
        pixel_values = batch["pixel_values"].to(ctx.device)
        labels = [{k: v.to(ctx.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = ctx.model(pixel_values=pixel_values, labels=labels)
        loss = outputs["loss"]

        ctx.optimizer.zero_grad()
        loss.backward()
        ctx.optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(ctx.train_loader)
    lr = ctx.optimizer.param_groups[0]["lr"]
    return avg_loss, lr


def save_results(
    ctx: TrainingContext,
    history: TrainingHistory,
    map_result: dict[str, Any],
    save_model: ModelSaver,
    logger: Any,
) -> None:
    """モデル保存 + レポート出力.

    Args:
        ctx: 学習コンテキスト.
        history: 学習履歴.
        map_result: 最後に検証されたエポックの mAP 計算結果.
        save_model: モデル保存関数.
        logger: ロガー.
    """
    last_dir = ctx.workspace_manager.get_last_dir()
    save_model(ctx, last_dir)
    logger.info(f"Last model saved to {last_dir}")

    history_path = ctx.workspace / "training_history.csv"
    history.save_csv(history_path)
    logger.info(f"Training history saved to {history_path}")

    loss_plotter = LossPlotter(history)
    metrics_plotter = MetricsPlotter(history)
    report_plotter = TrainingReportPlotter(loss_plotter, metrics_plotter)
    report_path = ctx.workspace / "training_report.html"
    report_plotter.plot(report_path)
    logger.info(f"Training report saved to {report_path}")

    if "precision" in map_result:
        pr_plotter = PRCurvePlotter(map_result["precision"])
        pr_path = ctx.workspace / "pr_curve.html"
        pr_plotter.plot(pr_path)
        logger.info(f"PR curve saved to {pr_path}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _save_best(
    ctx: TrainingContext,
    save_model: ModelSaver,
    metric_name: str,
    metric_value: float,
    logger: Any,
) -> None:
    """Best model を保存.

    Args:
        ctx: 学習コンテキスト.
        save_model: モデル保存関数.
        metric_name: メトリクス名 (ログ用).
        metric_value: メトリクス値 (ログ用).
        logger: ロガー.
    """
    best_dir = ctx.workspace_manager.get_best_dir()
    save_model(ctx, best_dir)
    logger.info(f"Best model saved to {best_dir} ({metric_name}: {metric_value:.4f})")
