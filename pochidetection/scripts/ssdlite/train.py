"""SSDLite MobileNetV3 学習スクリプト.

torchvision の SSDLite を COCO 形式データセットでファインチューニングする.
"""

from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torchmetrics.detection import MeanAveragePrecision

from pochidetection.datasets import SsdCocoDataset
from pochidetection.logging import LoggerManager
from pochidetection.models import SSDLiteModel
from pochidetection.scripts.common.training import (
    TrainingContext,
    build_data_loaders,
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

    dataset_factory = partial(SsdCocoDataset, image_size=image_size)
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

    # BN の running statistics を退避 (train モード切替による汚染を防止)
    bn_states = _save_bn_states(ctx.model)

    with torch.no_grad():
        for batch in ctx.val_loader:
            pixel_values = batch["pixel_values"].to(ctx.device)
            labels = [
                {k: v.to(ctx.device) for k, v in t.items()} for t in batch["labels"]
            ]

            # 学習モードで loss を計算 (torchvision SSD は train モード必須)
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

    # BN の running statistics を復元
    _restore_bn_states(ctx.model, bn_states)

    avg_val_loss = val_loss / len(ctx.val_loader)
    map_result = ctx.map_metric.compute()
    return avg_val_loss, map_result


def _save_bn_states(model: nn.Module) -> dict[str, torch.Tensor]:
    """BN の running statistics を退避する.

    Args:
        model: 対象モデル.

    Returns:
        モジュール名をキー, running statistics のクローンを値とする辞書.
    """
    states: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            if module.running_mean is not None:
                states[f"{name}.running_mean"] = module.running_mean.clone()
            if module.running_var is not None:
                states[f"{name}.running_var"] = module.running_var.clone()
            if module.num_batches_tracked is not None:
                states[f"{name}.num_batches_tracked"] = (
                    module.num_batches_tracked.clone()
                )
    return states


def _restore_bn_states(model: nn.Module, states: dict[str, torch.Tensor]) -> None:
    """退避した BN の running statistics を復元する.

    Args:
        model: 対象モデル.
        states: _save_bn_states で退避した辞書.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            if f"{name}.running_mean" in states:
                module.running_mean.copy_(states[f"{name}.running_mean"])
            if f"{name}.running_var" in states:
                module.running_var.copy_(states[f"{name}.running_var"])
            if f"{name}.num_batches_tracked" in states:
                module.num_batches_tracked.copy_(states[f"{name}.num_batches_tracked"])
