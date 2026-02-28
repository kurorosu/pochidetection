"""RT-DETR 学習スクリプト.

transformersのRT-DETRをCOCO形式データセットでファインチューニングする.
"""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from transformers import RTDetrImageProcessor

from pochidetection.core import DetectionCollator
from pochidetection.datasets import CocoDetectionDataset
from pochidetection.logging import LoggerManager
from pochidetection.models import RTDetrModel
from pochidetection.utils import TrainingHistory, WorkspaceManager
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
    # ロガー設定
    logger_manager = LoggerManager()
    logger = logger_manager.get_logger(__name__)

    # 設定値を取得
    device = config["device"]
    num_classes = config["num_classes"]
    model_name = config["model_name"]
    image_size = config.get("image_size", {"height": 640, "width": 640})
    data_root = Path(config["data_root"])
    train_dir = data_root / config["train_split"]
    val_dir = data_root / config["val_split"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]

    # ワークスペース作成
    workspace_manager = WorkspaceManager(config["work_dir"])
    workspace = workspace_manager.create_workspace()
    workspace_manager.save_config(config_path)

    logger.info(f"Device: {device}")
    logger.info(f"Num classes: {num_classes}")
    logger.info(f"Image size: {image_size}")
    logger.info(f"Workspace: {workspace}")

    # モデルとプロセッサ
    processor = RTDetrImageProcessor.from_pretrained(model_name, size=image_size)
    model = RTDetrModel(model_name, num_classes=num_classes)
    model.to(device)

    # データローダー
    train_dataset = CocoDetectionDataset(train_dir, processor)
    val_dataset = CocoDetectionDataset(val_dir, processor)

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

    # オプティマイザ
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # mAP計算用
    # RT-DETRは300クエリを出力するため警告が出るが, 閾値はデフォルト[1,10,100]を維持
    # extended_summary=True でPR曲線用のprecision/recallデータを取得
    map_metric = MeanAveragePrecision(iou_type="bbox", extended_summary=True)
    map_metric.warn_on_many_detections = False

    # ベストmAP追跡用
    best_map = 0.0

    # 学習履歴
    history = TrainingHistory()

    # 学習ループ
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            # 順伝播
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs["loss"]

            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        lr = optimizer.param_groups[0]["lr"]

        # 検証
        model.eval()
        val_loss = 0.0
        map_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = [
                    {k: v.to(device) for k, v in t.items()} for t in batch["labels"]
                ]
                # 内部モデルを直接呼び出してtransformersの出力オブジェクトを取得
                outputs = model.model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

                # mAP計算用: 公式APIで後処理
                results = processor.post_process_object_detection(
                    outputs, threshold=0.2, target_sizes=None
                )

                for i, result in enumerate(results):
                    # 予測 (正規化座標のまま)
                    pred_boxes_xyxy = result["boxes"]
                    pred_scores = result["scores"]
                    pred_labels_filtered = result["labels"]

                    # ターゲット (cxcywh -> xyxy)
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

                    # torchmetrics形式に変換
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
                    map_metric.update(preds, targets)

        avg_val_loss = val_loss / len(val_loader)
        map_result = map_metric.compute()
        mAP = map_result["map"].item()
        mAP_50 = map_result["map_50"].item()
        mAP_75 = map_result["map_75"].item()

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {avg_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"mAP: {mAP:.4f}, "
            f"mAP@50: {mAP_50:.4f}, "
            f"mAP@75: {mAP_75:.4f}, "
            f"LR: {lr:.2e}"
        )

        # 履歴に追加
        history.add(
            epoch=epoch + 1,
            train_loss=avg_loss,
            val_loss=avg_val_loss,
            mAP=mAP,
            mAP_50=mAP_50,
            mAP_75=mAP_75,
            lr=lr,
        )

        # ベストモデル保存
        if mAP > best_map:
            best_map = mAP
            best_dir = workspace_manager.get_best_dir()
            model.model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            logger.info(f"Best model saved to {best_dir} (mAP: {best_map:.4f})")

    # 最終モデル保存
    last_dir = workspace_manager.get_last_dir()
    model.model.save_pretrained(last_dir)
    processor.save_pretrained(last_dir)
    logger.info(f"Last model saved to {last_dir}")

    # 学習履歴を保存
    history_path = workspace / "training_history.csv"
    history.save_csv(history_path)
    logger.info(f"Training history saved to {history_path}")

    # 学習レポートを出力
    loss_plotter = LossPlotter(history)
    metrics_plotter = MetricsPlotter(history)
    report_plotter = TrainingReportPlotter(loss_plotter, metrics_plotter)
    report_path = workspace / "training_report.html"
    report_plotter.plot(report_path)
    logger.info(f"Training report saved to {report_path}")

    # PR曲線を出力 (最終epochの検証結果から)
    if "precision" in map_result:
        pr_plotter = PRCurvePlotter(map_result["precision"])
        pr_path = workspace / "pr_curve.html"
        pr_plotter.plot(pr_path)
        logger.info(f"PR curve saved to {pr_path}")
