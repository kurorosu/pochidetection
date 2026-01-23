"""RT-DETR クイックスタート.

transformersのRT-DETRをCOCO形式データセットでファインチューニングし、推論する最小構成スクリプト.

使用方法:
    uv run python quickstart.py
"""

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# =============================================================================
# 設定
# =============================================================================
DATA_ROOT = Path("data")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
NUM_CLASSES = 1  # (背景クラスは含まない)
BATCH_SIZE = 2
EPOCHS = 50  # 小さいデータセットなので多めに
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# データセット
# =============================================================================
class CocoDataset(Dataset):
    """COCO形式データセット."""

    def __init__(self, root: Path, processor: RTDetrImageProcessor) -> None:
        """COCO形式データセットを初期化."""
        self.root = root
        self.processor = processor

        # アノテーション読み込み (instances_*.json または annotations.json)
        ann_file: Path | None = None
        for pattern in ["instances_*.json", "annotations.json"]:
            matches = list(root.glob(pattern))
            if matches:
                ann_file = matches[0]
                break
        if ann_file is None:
            raise FileNotFoundError(f"Annotation file not found in {root}")
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.images = data["images"]
        self.categories = [
            c
            for c in data["categories"]
            if c["name"].lower() not in {"_background_", "background"}
        ]

        # カテゴリIDを0始まりの連続インデックスにマッピング
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(self.categories)}

        # image_id -> annotations のマッピング
        self.annotations: dict[int, list[dict[str, Any]]] = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

    def __len__(self) -> int:
        """データセットのサンプル数を返す."""
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """インデックスでサンプルを取得."""
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = self.root / img_info["file_name"]

        # 画像読み込み
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # アノテーション取得
        anns = self.annotations.get(img_id, [])

        # COCO形式 [x, y, w, h] -> 正規化 [cx, cy, w, h]
        boxes = []
        labels = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in self.cat_id_to_idx:
                continue  # 背景クラスはスキップ

            x, y, w, h = ann["bbox"]
            # 正規化された中心座標形式に変換
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            boxes.append([cx, cy, nw, nh])
            labels.append(self.cat_id_to_idx[cat_id])

        # 前処理
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        # ターゲット (RT-DETRの形式)
        target = {
            "boxes": (
                torch.tensor(boxes, dtype=torch.float32)
                if boxes
                else torch.zeros((0, 4))
            ),
            "class_labels": (
                torch.tensor(labels, dtype=torch.int64)
                if labels
                else torch.zeros((0,), dtype=torch.int64)
            ),
        }

        return {
            "pixel_values": pixel_values,
            "labels": target,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """バッチ化."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


# =============================================================================
# 学習
# =============================================================================
def train() -> None:
    """ファインチューニング."""
    print(f"Device: {DEVICE}")
    print(f"Num classes: {NUM_CLASSES}")

    # モデルとプロセッサ
    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
    model = RTDetrForObjectDetection.from_pretrained(
        "PekingU/rtdetr_r50vd",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)

    # データローダー
    train_dataset = CocoDataset(TRAIN_DIR, processor)
    val_dataset = CocoDataset(VAL_DIR, processor)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # オプティマイザ
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 学習ループ
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

            # 順伝播
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

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
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_loss:.4f}")

        # 検証
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                labels = [
                    {k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]
                ]
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Val Loss: {avg_val_loss:.4f}")

    # モデル保存
    output_dir = Path("work_dirs")
    output_dir.mkdir(exist_ok=True)
    model.save_pretrained(output_dir / "rtdetr_finetuned")
    processor.save_pretrained(output_dir / "rtdetr_finetuned")
    print(f"Model saved to {output_dir / 'rtdetr_finetuned'}")


# =============================================================================
# 推論
# =============================================================================
def infer(image_path: str, threshold: float = 0.5) -> None:
    """推論."""
    model_dir = Path("work_dirs/rtdetr_finetuned")

    if not model_dir.exists():
        print("Model not found. Please run training first.")
        return

    # モデル読み込み
    processor = RTDetrImageProcessor.from_pretrained(model_dir)
    model = RTDetrForObjectDetection.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()

    # 画像読み込み
    image = Image.open(image_path).convert("RGB")

    # 前処理
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 推論
    with torch.no_grad():
        outputs = model(**inputs)

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
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "infer":
        if len(sys.argv) < 3:
            print("Usage: python quickstart.py infer <image_path> [threshold]")
            sys.exit(1)
        image_path = sys.argv[2]
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        infer(image_path, threshold)
    else:
        train()
