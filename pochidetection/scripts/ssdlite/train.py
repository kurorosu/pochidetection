"""SSDLite MobileNetV3 学習スクリプト.

torchvision の SSDLite を COCO 形式データセットでファインチューニングする.
"""

from pathlib import Path
from typing import Any, Literal, NamedTuple

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from pochidetection.core import DetectionCollator
from pochidetection.datasets import SsdCocoDataset
from pochidetection.logging import LoggerManager
from pochidetection.models import SSDLiteModel
from pochidetection.utils import (
    EarlyStopping,
    TrainingHistory,
    WorkspaceManager,
    build_scheduler,
)
from pochidetection.visualization import (
    LossPlotter,
    MetricsPlotter,
    PRCurvePlotter,
    TrainingReportPlotter,
)


def train(config: dict[str, Any], config_path: str) -> None:
    """ファインチューニング.

    Args:
        config: 設定辞書.
        config_path: 設定ファイルのパス (ワークスペースにコピーするため).
    """
    logger = LoggerManager().get_logger(__name__)

    ctx = _setup_training(config, config_path, logger)

    best_map = 0.0
    history = TrainingHistory()
    map_result: dict[str, Any] = {}

    early_stopping = _build_early_stopping(config)
    if early_stopping is not None:
        logger.info(
            f"Early Stopping: patience={early_stopping.patience}, "
            f"metric={config['early_stopping_metric']}, "
            f"min_delta={config['early_stopping_min_delta']}"
        )

    for epoch in range(ctx.epochs):
        avg_loss, lr = _train_one_epoch(ctx, logger)
        avg_val_loss, map_result = _validate(ctx, logger)

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
                _save_best_model(ctx, metric, value, logger)

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
                _save_best_model(ctx, "mAP", mAP, logger)

    _save_results(ctx, history, map_result, logger)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _save_best_model(
    ctx: "_TrainingContext",
    metric_name: str,
    metric_value: float,
    logger: Any,
) -> None:
    """Best model を保存.

    Args:
        ctx: 学習コンテキスト.
        metric_name: メトリクス名 (ログ用).
        metric_value: メトリクス値 (ログ用).
        logger: ロガー.
    """
    best_dir = ctx.workspace_manager.get_best_dir()
    torch.save(ctx.model.model.state_dict(), best_dir / "model.pth")
    logger.info(f"Best model saved to {best_dir} ({metric_name}: {metric_value:.4f})")


def _build_early_stopping(config: dict[str, Any]) -> EarlyStopping | None:
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


class _TrainingContext(NamedTuple):
    """_setup_training の戻り値."""

    model: SSDLiteModel
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


def _setup_training(
    config: dict[str, Any],
    config_path: str,
    logger: Any,
) -> _TrainingContext:
    """学習環境の構築.

    Args:
        config: 設定辞書.
        config_path: 設定ファイルのパス.
        logger: ロガー.

    Returns:
        構築済みの学習コンテキスト.
    """
    device = config["device"]
    num_classes = config["num_classes"]
    image_size = config.get("image_size", {"height": 320, "width": 320})
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    train_score_threshold = config["train_score_threshold"]

    workspace_manager = WorkspaceManager(config["work_dir"])
    workspace = workspace_manager.create_workspace()
    workspace_manager.save_config(config_path)

    logger.info(f"Architecture: SSDLite MobileNetV3")
    logger.info(f"Device: {device}")
    logger.info(f"Num classes: {num_classes}")
    logger.info(f"Image size: {image_size}")
    logger.info(f"Workspace: {workspace}")

    model = SSDLiteModel(num_classes=num_classes)
    model.to(device)

    train_loader, val_loader = _build_data_loaders(config, image_size, logger)

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

    return _TrainingContext(
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


def _build_data_loaders(
    config: dict[str, Any],
    image_size: dict[str, int],
    logger: Any,
) -> tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
    """学習・検証用データローダーを構築.

    Args:
        config: 設定辞書.
        image_size: 画像サイズ.
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

    train_dataset = SsdCocoDataset(train_dir, image_size)
    val_dataset = SsdCocoDataset(val_dir, image_size)

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


def _train_one_epoch(
    ctx: _TrainingContext,
    logger: Any,
) -> tuple[float, float]:
    """1エポックの学習.

    Args:
        ctx: 学習コンテキスト.
        logger: ロガー.

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


def _validate(
    ctx: _TrainingContext,
    logger: Any,
) -> tuple[float, dict[str, Any]]:
    """検証ループ + mAP 計算.

    Args:
        ctx: 学習コンテキスト.
        logger: ロガー.

    Returns:
        (平均検証損失, mAP 計算結果辞書) のタプル.
    """
    ctx.model.eval()
    val_loss = 0.0
    ctx.map_metric.reset()

    with torch.no_grad():
        for batch in ctx.val_loader:
            pixel_values = batch["pixel_values"].to(ctx.device)
            labels = [
                {k: v.to(ctx.device) for k, v in t.items()} for t in batch["labels"]
            ]

            # 学習モードで loss を計算
            ctx.model.train()
            train_outputs = ctx.model(pixel_values=pixel_values, labels=labels)
            val_loss += train_outputs["loss"].item()

            # 推論モードで予測を取得
            ctx.model.eval()
            infer_outputs = ctx.model(pixel_values=pixel_values)

            for i, pred in enumerate(infer_outputs["predictions"]):
                # スコア閾値でフィルタ
                mask = pred["scores"] >= ctx.train_score_threshold
                pred_boxes = pred["boxes"][mask]
                pred_scores = pred["scores"][mask]
                pred_labels = pred["labels"][mask]

                # ターゲット: 1-indexed → 0-indexed
                target_boxes = labels[i]["boxes"]
                target_labels = labels[i]["labels"] - 1

                preds = [
                    {
                        "boxes": pred_boxes.cpu(),
                        "scores": pred_scores.cpu(),
                        "labels": pred_labels.cpu(),
                    }
                ]
                targets = [
                    {
                        "boxes": target_boxes.cpu(),
                        "labels": target_labels.cpu(),
                    }
                ]
                ctx.map_metric.update(preds, targets)

    avg_val_loss = val_loss / len(ctx.val_loader)
    map_result = ctx.map_metric.compute()
    return avg_val_loss, map_result


def _save_results(
    ctx: _TrainingContext,
    history: TrainingHistory,
    map_result: dict[str, Any],
    logger: Any,
) -> None:
    """モデル保存 + レポート出力.

    Args:
        ctx: 学習コンテキスト.
        history: 学習履歴.
        map_result: 最後に検証されたエポックの mAP 計算結果.
        logger: ロガー.
    """
    last_dir = ctx.workspace_manager.get_last_dir()
    torch.save(ctx.model.model.state_dict(), last_dir / "model.pth")
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
