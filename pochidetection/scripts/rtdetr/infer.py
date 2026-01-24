"""RT-DETR 推論スクリプト.

学習済みRT-DETRモデルで画像の物体検出を行う.
"""

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import RTDetrImageProcessor

from pochidetection.models import RTDetrModel
from pochidetection.utils import WorkspaceManager


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

    # モデル読み込み (保存時の image_size が使用される)
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
            f"  Label: {label.item()}, Score: {score.item():.3f}, "
            f"Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
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
