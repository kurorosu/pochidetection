"""RT-DETR クイックスタート.

transformersのRT-DETRをCOCO形式データセットでファインチューニングし、推論する最小構成スクリプト.

使用方法:
    uv run python quickstart.py
    uv run python quickstart.py configs/rtdetr_coco.py
"""

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
from pochidetection.models import RTDetrModel
from pochidetection.utils import ConfigLoader

# =============================================================================
# 設定読み込み
# =============================================================================
DEFAULT_CONFIG = "configs/rtdetr_coco.py"


# =============================================================================
# 学習
# =============================================================================
def train(config: dict[str, Any]) -> None:
    """ファインチューニング."""
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
    work_dir = Path(config["work_dir"])

    print(f"Device: {device}")
    print(f"Num classes: {num_classes}")

    # モデルとプロセッサ
    processor = RTDetrImageProcessor.from_pretrained(model_name)
    model = RTDetrModel(model_name, num_classes=num_classes)
    model.to(device)

    # データローダー
    train_dataset = CocoDetectionDataset(train_dir, processor)
    val_dataset = CocoDetectionDataset(val_dir, processor)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

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

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {avg_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"mAP: {mAP:.4f}, "
            f"mAP@50: {mAP_50:.4f}, "
            f"LR: {lr:.2e}"
        )

    # モデル保存
    work_dir.mkdir(exist_ok=True)
    model.model.save_pretrained(work_dir / "rtdetr_finetuned")
    processor.save_pretrained(work_dir / "rtdetr_finetuned")
    print(f"Model saved to {work_dir / 'rtdetr_finetuned'}")


# =============================================================================
# 推論
# =============================================================================
def infer(config: dict[str, Any], image_path: str, threshold: float = 0.5) -> None:
    """推論."""
    device = config["device"]
    work_dir = Path(config["work_dir"])
    model_dir = work_dir / "rtdetr_finetuned"

    if not model_dir.exists():
        print("Model not found. Please run training first.")
        return

    # モデル読み込み
    processor = RTDetrImageProcessor.from_pretrained(model_dir)
    model = RTDetrModel(str(model_dir))
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
# メイン
# =============================================================================
if __name__ == "__main__":
    # 設定ファイルを読み込み
    config_path = (
        sys.argv[1]
        if len(sys.argv) > 1 and not sys.argv[1] == "infer"
        else DEFAULT_CONFIG
    )
    config = ConfigLoader.load(config_path)

    if len(sys.argv) > 1 and sys.argv[1] == "infer":
        if len(sys.argv) < 3:
            print("Usage: python quickstart.py infer <image_path> [threshold]")
            sys.exit(1)
        image_path = sys.argv[2]
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        infer(config, image_path, threshold)
    else:
        train(config)
