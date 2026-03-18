"""SSD 系モデル共通の検証ロジック・BN ヘルパー.

SSDLite / SSD300 の学習スクリプトで共有する検証ループと
BatchNorm running statistics の退避・復元ユーティリティを提供する.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from pochidetection.scripts.common.training import TrainingContext


def ssd_validate(
    ctx: TrainingContext,
    logger: logging.Logger,
) -> tuple[float, dict[str, Any]]:
    """SSD 系モデル共通の検証ループ + mAP 計算.

    torchvision SSD は loss 計算に train モードが必要なため,
    BN の running statistics を退避・復元して汚染を防止する.

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
    bn_states = save_bn_states(ctx.model)

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
    restore_bn_states(ctx.model, bn_states)

    avg_val_loss = val_loss / len(ctx.val_loader)
    map_result = ctx.map_metric.compute()
    return avg_val_loss, map_result


def save_bn_states(model: nn.Module) -> dict[str, torch.Tensor]:
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


def restore_bn_states(model: nn.Module, states: dict[str, torch.Tensor]) -> None:
    """退避した BN の running statistics を復元する.

    Args:
        model: 対象モデル.
        states: save_bn_states で退避した辞書.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            if f"{name}.running_mean" in states:
                module.running_mean.copy_(states[f"{name}.running_mean"])
            if f"{name}.running_var" in states:
                module.running_var.copy_(states[f"{name}.running_var"])
            if f"{name}.num_batches_tracked" in states:
                module.num_batches_tracked.copy_(states[f"{name}.num_batches_tracked"])
