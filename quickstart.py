"""RT-DETR クイックスタート.

transformersのRT-DETRをCOCO形式データセットでファインチューニングし、推論する最小構成スクリプト.

使用方法:
    uv run python quickstart.py
    uv run python quickstart.py configs/rtdetr_coco.py
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from transformers import RTDetrImageProcessor

from pochidetection.core import DetectionCollator
from pochidetection.datasets import CocoDetectionDataset
from pochidetection.logging import LoggerManager, LogLevel
from pochidetection.models import RTDetrModel
from pochidetection.utils import ConfigLoader, WorkspaceManager

# =============================================================================
# 設定読み込み
# =============================================================================
DEFAULT_CONFIG = "configs/rtdetr_coco.py"


# =============================================================================
# 学習
# =============================================================================
def train(config: dict[str, Any], config_path: str) -> None:
    """ファインチューニング.

    Args:
        config: 設定辞書.
        config_path: 設定ファイルのパス (ワークスペースにコピーするため).
    """
    # ロガー設定
    logger_manager = LoggerManager()
    logger = logger_manager.get_logger(__name__, level=LogLevel.INFO)

    # 設定値を取得
    device = config["device"]
    num_classes = config["num_classes"]
    model_name = config["model_name"]
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
    logger.info(f"Workspace: {workspace}")

    # モデルとプロセッサ
    processor = RTDetrImageProcessor.from_pretrained(model_name)
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
    map_metric = MeanAveragePrecision(iou_type="bbox")
    map_metric.warn_on_many_detections = False

    # ベストmAP追跡用
    best_map = 0.0

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


# =============================================================================
# 推論
# =============================================================================
def infer(
    config: dict[str, Any],
    image_path: str,
    threshold: float = 0.5,
    model_dir: str | None = None,
) -> None:
    """推論.

    Args:
        config: 設定辞書.
        image_path: 推論対象の画像パス.
        threshold: 検出信頼度閾値.
        model_dir: モデルディレクトリ. Noneの場合は最新ワークスペースのbestを使用.
    """
    device = config["device"]

    # モデルディレクトリの決定
    if model_dir is not None:
        model_path = Path(model_dir)
        if not model_path.exists():
            print(f"Model not found at {model_path}")
            return
    else:
        # 最新のワークスペースからベストモデルを探す
        workspace_manager = WorkspaceManager(config["work_dir"])
        workspaces = workspace_manager.get_available_workspaces()

        if not workspaces:
            print("No trained models found. Please run training first.")
            return

        # 最新のワークスペースのbestディレクトリを使用
        latest_workspace = Path(str(workspaces[-1]["path"]))
        model_path = latest_workspace / "best"

        if not model_path.exists():
            print(f"Best model not found at {model_path}. Please run training first.")
            return

    print(f"Loading model from {model_path}")

    # モデル読み込み
    processor = RTDetrImageProcessor.from_pretrained(model_path)
    model = RTDetrModel(str(model_path))
    model.to(device)
    model.eval()

    # 画像読み込み
    image = Image.open(image_path).convert("RGB")

    # 前処理
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推論
    with torch.no_grad():
        outputs = model.model(**inputs)

    # 後処理 (transformers公式メソッド)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([image.size[::-1]]),  # (height, width)
        threshold=threshold,
    )[0]

    print(f"Detected {len(results['boxes'])} objects:")
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = box.tolist()
        print(
            f"  Label: {label.item()}, Score: {score.item():.3f}, Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
        )

    # 検出結果を画像に描画
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # 画像サイズに応じた線の太さとフォントサイズ
    base_size = max(width, height)
    line_width = max(2, int(base_size / 300))
    font_size = max(12, int(base_size / 40))

    # フォント設定 (デフォルトフォントを使用)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = box.tolist()
        x1, y1, x2, y2 = box

        # ボックス描画
        draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)

        # ラベルと信頼度のテキスト
        text = f"{label.item()}: {score.item():.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # テキスト背景 (ボックス左上)
        padding = 2
        bg_x1 = x1
        bg_y1 = max(0, y1 - text_height - padding * 2)
        bg_x2 = x1 + text_width + padding * 2
        bg_y2 = y1
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill="red")

        # テキスト描画 (白文字)
        draw.text((x1 + padding, bg_y1 + padding), text, fill="white", font=font)

    # 結果画像を保存
    input_path = Path(image_path)
    output_path = Path(f"{input_path.stem}_result{input_path.suffix}")
    image.save(output_path)
    print(f"Result saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース.

    Returns:
        パースされた引数.
    """
    parser = argparse.ArgumentParser(
        description="RT-DETR ファインチューニング・推論スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  学習:
    uv run python quickstart.py
    uv run python quickstart.py -c configs/rtdetr_coco.py

  推論:
    uv run python quickstart.py infer -i image.jpg
    uv run python quickstart.py infer -i image.jpg -t 0.3
    uv run python quickstart.py infer -i image.jpg -m work_dirs/20260124_001/best
        """,
    )

    # サブコマンド
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")

    # 学習コマンド (デフォルト)
    train_parser = subparsers.add_parser("train", help="モデルの学習")
    train_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"設定ファイルのパス (default: {DEFAULT_CONFIG})",
    )

    # 推論コマンド
    infer_parser = subparsers.add_parser("infer", help="画像の推論")
    infer_parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="推論対象の画像パス",
    )
    infer_parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="検出信頼度閾値 (default: 0.5)",
    )
    infer_parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        default=None,
        help="モデルディレクトリ (default: 最新ワークスペースのbest)",
    )
    infer_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"設定ファイルのパス (default: {DEFAULT_CONFIG})",
    )

    # 引数なしの場合はtrainをデフォルトに
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"設定ファイルのパス (default: {DEFAULT_CONFIG})",
    )

    return parser.parse_args()


# =============================================================================
# メイン
# =============================================================================
if __name__ == "__main__":
    args = parse_args()

    # コマンドに応じて処理を分岐
    if args.command == "infer":
        config = ConfigLoader.load(args.config)
        infer(config, args.image, args.threshold, args.model_dir)
    else:
        # train または コマンド未指定 (デフォルト)
        config_path = args.config
        config = ConfigLoader.load(config_path)
        train(config, config_path)
