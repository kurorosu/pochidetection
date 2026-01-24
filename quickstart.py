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
from PIL import Image
from torch.utils.data import DataLoader
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

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}")

        # 検証
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = [
                    {k: v.to(device) for k, v in t.items()} for t in batch["labels"]
                ]
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs["loss"].item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss:.4f}")

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
