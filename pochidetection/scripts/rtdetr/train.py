"""RT-DETR 学習スクリプト.

transformersのRT-DETRをCOCO形式データセットでファインチューニングする.
"""

import logging
from typing import Any

import torch
from transformers import RTDetrImageProcessor

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.datasets import CocoDetectionDataset
from pochidetection.interfaces.model import IDetectionModel
from pochidetection.logging import LoggerManager
from pochidetection.models import RTDetrModel
from pochidetection.training.loop import (
    TrainingContext,
    TrainingLoop,
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
    TrainingLoop(config, ctx, _validate).run()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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
    # processor はモデル構築後に取得する必要があるため, リストで共有する
    processor_holder: list[RTDetrImageProcessor] = []

    def model_factory(cfg: DetectionConfigDict) -> IDetectionModel:
        model = RTDetrModel(
            cfg["model_name"],
            num_classes=cfg["num_classes"],
            image_size=cfg["image_size"],
        )
        processor_holder.append(model.processor)
        return model

    def dataset_factory(path: Any) -> Any:
        image_size = config["image_size"]
        return CocoDetectionDataset(
            path,
            processor=processor_holder[0],
            letterbox=config.get("letterbox", True),
            image_size=(image_size["height"], image_size["width"]),
        )

    return setup_training(
        config=config,
        config_path=config_path,
        model_factory=model_factory,
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
    model = ctx.model
    if not isinstance(model, RTDetrModel):
        raise TypeError(f"Expected RTDetrModel, got {type(model).__name__}")

    processor = model.processor

    model.eval()
    val_loss = 0.0
    ctx.map_metric.reset()

    with torch.no_grad():
        for batch in ctx.val_loader:
            pixel_values = batch["pixel_values"].to(ctx.device)
            labels = [
                {k: v.to(ctx.device) for k, v in t.items()} for t in batch["labels"]
            ]
            outputs = model.model(pixel_values=pixel_values, labels=labels)
            val_loss += outputs.loss.item()

            results = processor.post_process_object_detection(
                outputs,
                threshold=ctx.train_score_threshold,
                target_sizes=None,
            )

            for i, result in enumerate(results):
                pred_boxes_xyxy = result["boxes"]
                pred_scores = result["scores"]
                pred_labels_filtered = result["labels"]

                target_boxes = labels[i]["boxes"]
                target_labels = labels[i]["class_labels"]
                if target_boxes.numel() > 0:
                    tcx, tcy, tw, th = target_boxes.unbind(-1)
                    target_boxes_xyxy = torch.stack(
                        [tcx - tw / 2, tcy - th / 2, tcx + tw / 2, tcy + th / 2],
                        dim=-1,
                    )
                else:
                    target_boxes_xyxy = target_boxes

                preds = [
                    {
                        "boxes": pred_boxes_xyxy.cpu(),
                        "scores": pred_scores.cpu(),
                        "labels": pred_labels_filtered.cpu(),
                    }
                ]
                targets = [
                    {
                        "boxes": target_boxes_xyxy.cpu(),
                        "labels": target_labels.cpu(),
                    }
                ]
                ctx.map_metric.update(preds, targets)

    avg_val_loss = val_loss / len(ctx.val_loader)
    map_result = ctx.map_metric.compute()
    return avg_val_loss, map_result
