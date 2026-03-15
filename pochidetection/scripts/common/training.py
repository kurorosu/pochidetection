"""学習ループの共通ロジック.

RT-DETR と SSDLite で共有されるエポックループ, Early Stopping,
データローダー構築, レポート出力のロジックを提供する.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.core import DetectionCollator
from pochidetection.interfaces.model import IDetectionModel
from pochidetection.logging import LoggerManager
from pochidetection.scripts.common.inference import setup_cudnn_benchmark
from pochidetection.utils import (
    EarlyStopping,
    TrainingHistory,
    WorkspaceManager,
    build_scheduler,
)
from pochidetection.visualization import (
    F1ConfidencePlotter,
    LossPlotter,
    MetricsPlotter,
    PRCurvePlotter,
    TrainingReportPlotter,
)


class TrainingContext(NamedTuple):
    """学習コンテキスト (アーキテクチャ共通)."""

    model: IDetectionModel
    train_loader: DataLoader[dict[str, Any]]
    val_loader: DataLoader[dict[str, Any]]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    map_metric: MeanAveragePrecision
    workspace: Path
    workspace_manager: WorkspaceManager
    device: str
    epochs: int
    train_score_threshold: float


DatasetFactory = Callable[[Path], Dataset[dict[str, Any]]]
ModelFactory = Callable[[DetectionConfigDict], IDetectionModel]


def setup_training(
    config: DetectionConfigDict,
    config_path: str,
    model_factory: ModelFactory,
    dataset_factory: DatasetFactory,
    logger: logging.Logger,
) -> TrainingContext:
    """学習環境の共通セットアップ.

    ワークスペース作成, モデル構築, データローダー構築, オプティマイザ,
    スケジューラ, mAP メトリクスの初期化を共通化する.

    Args:
        config: 設定辞書.
        config_path: 設定ファイルのパス.
        model_factory: config を受け取りモデルを返すファクトリ.
        dataset_factory: ディレクトリパスを受け取りデータセットを返すファクトリ.
        logger: ロガー.

    Returns:
        構築済みの学習コンテキスト.
    """
    setup_cudnn_benchmark(config)

    device = config["device"]
    image_size = config.get("image_size", {"height": 640, "width": 640})
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    train_score_threshold = config["train_score_threshold"]

    workspace_manager = WorkspaceManager(config["work_dir"])
    workspace = workspace_manager.create_workspace()
    workspace_manager.save_config(config, Path(config_path).name)

    logger.info(f"Device: {device}")
    logger.info(f"Num classes: {config['num_classes']}")
    logger.info(f"Image size: {image_size}")
    logger.info(f"Workspace: {workspace}")

    model = model_factory(config)
    model.to(device)

    train_loader, val_loader = build_data_loaders(config, dataset_factory, logger)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=config.get("lr_scheduler"),
        scheduler_params=config.get("lr_scheduler_params"),
        epochs=epochs,
    )
    if scheduler is not None:
        logger.info(f"LR Scheduler: {scheduler.__class__.__name__}")

    map_metric = MeanAveragePrecision(iou_type="bbox", extended_summary=True)
    map_metric.warn_on_many_detections = False

    return TrainingContext(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        map_metric=map_metric,
        workspace=workspace,
        workspace_manager=workspace_manager,
        device=device,
        epochs=epochs,
        train_score_threshold=train_score_threshold,
    )


def build_data_loaders(
    config: DetectionConfigDict,
    dataset_factory: DatasetFactory,
    logger: logging.Logger,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    """学習・検証用データローダーを構築.

    Args:
        config: 設定辞書.
        dataset_factory: ディレクトリパスを受け取りデータセットを返すファクトリ.
        logger: ロガー.

    Returns:
        (train_loader, val_loader) のタプル.

    Raises:
        ValueError: データローダーが空の場合.
    """
    data_root = Path(config["data_root"])
    train_dir = data_root / config["train_split"]
    val_dir = data_root / config["val_split"]
    batch_size = config["batch_size"]

    train_dataset = dataset_factory(train_dir)
    val_dataset = dataset_factory(val_dir)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    collator = DetectionCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError(
            f"DataLoader が空です (train: {len(train_loader)} batches, "
            f"val: {len(val_loader)} batches). "
            f"データセットまたはバッチサイズを確認してください."
        )

    return train_loader, val_loader


class Validator(Protocol):
    """検証ループのプロトコル."""

    def __call__(
        self, ctx: TrainingContext, logger: logging.Logger
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
    config: DetectionConfigDict,
    ctx: TrainingContext,
    validate: Validator,
) -> None:
    """共通学習ループ.

    Args:
        config: 設定辞書.
        ctx: 学習コンテキスト.
        validate: 検証関数.
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

        cur_map = map_result["map"].item()
        cur_map_50 = map_result["map_50"].item()
        cur_map_75 = map_result["map_75"].item()

        logger.info(
            f"Epoch {epoch + 1}/{ctx.epochs} - "
            f"Train Loss: {avg_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"mAP: {cur_map:.4f}, "
            f"mAP@50: {cur_map_50:.4f}, "
            f"mAP@75: {cur_map_75:.4f}, "
            f"LR: {lr:.2e}"
        )

        history.add(
            epoch=epoch + 1,
            train_loss=avg_loss,
            val_loss=avg_val_loss,
            map=cur_map,
            map_50=cur_map_50,
            map_75=cur_map_75,
            lr=lr,
        )

        if ctx.scheduler is not None:
            if isinstance(ctx.scheduler, ReduceLROnPlateau):
                ctx.scheduler.step(avg_val_loss)
            else:
                ctx.scheduler.step()

        if early_stopping is not None:
            metric = config["early_stopping_metric"]
            value = cur_map if metric == "mAP" else avg_val_loss
            should_stop = early_stopping.step(value, epoch + 1)

            if early_stopping.counter == 0:
                _save_best(ctx, metric, value, logger)

            if should_stop:
                logger.info(
                    f"Early Stopping: {early_stopping.patience} エポック連続で "
                    f"{metric} が改善しなかったため学習を終了します "
                    f"(best epoch: {early_stopping.best_epoch}, "
                    f"best {metric}: {early_stopping.best_value:.4f})"
                )
                break
        else:
            if cur_map > best_map:
                best_map = cur_map
                _save_best(ctx, "mAP", cur_map, logger)

    save_results(config, ctx, history, map_result, logger)


def build_early_stopping(config: DetectionConfigDict) -> EarlyStopping | None:
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
    config: DetectionConfigDict,
    ctx: TrainingContext,
    history: TrainingHistory,
    map_result: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """モデル保存 + レポート出力.

    Args:
        config: 設定辞書.
        ctx: 学習コンテキスト.
        history: 学習履歴.
        map_result: 最後に検証されたエポックの mAP 計算結果.
        logger: ロガー.
    """
    last_dir = ctx.workspace_manager.get_last_dir()
    ctx.model.save(last_dir)
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

    class_names = config.get("class_names")

    if "precision" in map_result:
        pr_plotter = PRCurvePlotter(map_result["precision"], class_names=class_names)
        pr_path = ctx.workspace / "pr_curve.html"
        pr_plotter.plot(pr_path)
        logger.info(f"PR curve saved to {pr_path}")

    if "precision" in map_result and "scores" in map_result:
        f1_plotter = F1ConfidencePlotter(
            map_result["precision"],
            map_result["scores"],
            class_names=class_names,
        )
        f1_path = ctx.workspace / "f1_confidence.html"
        f1_plotter.plot(f1_path)
        logger.info(f"F1-Confidence curve saved to {f1_path}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _save_best(
    ctx: TrainingContext,
    metric_name: str,
    metric_value: float,
    logger: logging.Logger,
) -> None:
    """Best model を保存.

    Args:
        ctx: 学習コンテキスト.
        metric_name: メトリクス名 (ログ用).
        metric_value: メトリクス値 (ログ用).
        logger: ロガー.
    """
    best_dir = ctx.workspace_manager.get_best_dir()
    ctx.model.save(best_dir)
    logger.info(f"Best model saved to {best_dir} ({metric_name}: {metric_value:.4f})")
