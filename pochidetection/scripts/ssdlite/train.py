"""SSDLite MobileNetV3 学習スクリプト.

torchvision の SSDLite を COCO 形式データセットでファインチューニングする.
"""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from pochidetection.core import DetectionCollator
from pochidetection.datasets import SsdCocoDataset
from pochidetection.logging import LoggerManager
from pochidetection.models import SSDLiteModel
from pochidetection.scripts.common.training import (
    TrainingContext,
    run_training_loop,
)
from pochidetection.utils import (
    WorkspaceManager,
    build_scheduler,
)


def train(config: dict[str, Any], config_path: str) -> None:
    """ファインチューニング.

    Args:
        config: 設定辞書.
        config_path: 設定ファイルのパス (ワークスペースにコピーするため).
    """
    logger = LoggerManager().get_logger(__name__)
    ctx = _setup_training(config, config_path, logger)
    run_training_loop(config, ctx, _validate)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _setup_training(
    config: dict[str, Any],
    config_path: str,
    logger: Any,
) -> TrainingContext:
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


def _validate(
    ctx: TrainingContext,
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

                target_boxes = labels[i]["boxes"]
                target_labels = labels[i]["class_labels"]

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
