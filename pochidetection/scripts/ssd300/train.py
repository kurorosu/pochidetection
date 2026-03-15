"""SSD300 VGG16 学習スクリプト.

torchvision の SSD300 を COCO 形式データセットでファインチューニングする.
"""

import logging
from functools import partial
from typing import Any

import torch
import torch.nn as nn

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.datasets import SsdCocoDataset
from pochidetection.interfaces.model import IDetectionModel
from pochidetection.logging import LoggerManager
from pochidetection.models import SSD300Model
from pochidetection.scripts.common.training import (
    TrainingContext,
    run_training_loop,
    setup_training,
)


def train(config: DetectionConfigDict, config_path: str) -> None:
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


def _create_model(config: DetectionConfigDict) -> IDetectionModel:
    """モデル固有の SSD300 インスタンスを構築する.

    Args:
        config: 設定辞書.

    Returns:
        構築済みの SSD300Model.
    """
    num_classes = config["num_classes"]
    nms_iou_threshold = config.get("nms_iou_threshold", 0.5)
    return SSD300Model(num_classes=num_classes, nms_iou_threshold=nms_iou_threshold)


def _setup_training(
    config: DetectionConfigDict,
    config_path: str,
    logger: logging.Logger,
) -> TrainingContext:
    """学習環境の構築.

    Args:
        config: 設定辞書.
        config_path: 設定ファイルのパス.
        logger: ロガー.

    Returns:
        構築済みの学習コンテキスト.
    """
    logger.info("Architecture: SSD300 VGG16")

    image_size = config["image_size"]
    dataset_factory = partial(SsdCocoDataset, image_size=image_size)

    return setup_training(
        config=config,
        config_path=config_path,
        model_factory=_create_model,
        dataset_factory=dataset_factory,
        logger=logger,
    )


def _validate(
    ctx: TrainingContext,
    logger: logging.Logger,
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
