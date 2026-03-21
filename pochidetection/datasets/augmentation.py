"""Data Augmentation パイプライン.

config の augmentation セクションから torchvision.transforms.v2 の
Compose パイプラインを構築する.
"""

from typing import Any

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from pochidetection.configs.schemas import AugmentationConfig
from pochidetection.logging import LoggerManager

logger = LoggerManager().get_logger(__name__)

# RandomApply でラップすべき変換 (自前で p パラメータを持たない)
_NEEDS_RANDOM_APPLY = {
    "ColorJitter",
    "GaussianBlur",
    "RandomGrayscale",
    "RandomAutocontrast",
    "RandomEqualize",
    "RandomPosterize",
    "RandomSolarize",
    "RandomErasing",
}

# 自前で p パラメータを持つ変換
_HAS_OWN_P = {
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomPhotometricDistort",
}


def build_augmentation(config: AugmentationConfig) -> v2.Compose | None:
    """Config から augmentation パイプラインを構築する.

    Args:
        config: AugmentationConfig インスタンス.

    Returns:
        構築済みの v2.Compose. transforms が空または enabled=False の場合は None.
    """
    if not config.enabled or not config.transforms:
        return None

    transforms: list[Any] = []

    for t_config in config.transforms:
        name = t_config.name
        p = t_config.p

        # name 以外の追加パラメータを取得
        extra = t_config.model_extra or {}

        # torchvision.transforms.v2 からクラスを取得
        if not hasattr(v2, name):
            logger.warning(f"Unknown transform: {name} (skipped)")
            continue

        transform_cls = getattr(v2, name)

        if name in _HAS_OWN_P:
            # 自前で p を持つ変換にはそのまま p を渡す
            transform = transform_cls(p=p, **extra)
        elif name in _NEEDS_RANDOM_APPLY and p < 1.0:
            # p < 1.0 の場合は RandomApply でラップ
            transform = v2.RandomApply([transform_cls(**extra)], p=p)
        else:
            transform = transform_cls(**extra)

        transforms.append(transform)

    if not transforms:
        return None

    logger.info(f"Augmentation: {len(transforms)} transforms enabled")
    return v2.Compose(transforms)


def apply_augmentation(
    augmentation: v2.Compose,
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """画像と bbox に augmentation を適用する.

    Args:
        augmentation: 構築済みの augmentation パイプライン.
        image: PIL 画像 (RGB).
        boxes: bbox テンソル (N, 4), COCO 形式 [x, y, w, h].
        labels: ラベルテンソル (N,).

    Returns:
        (変換後の PIL 画像, 変換後の boxes, 変換後の labels) のタプル.
        面積ゼロの bbox は除外される.
    """
    w, h = image.size

    # COCO [x, y, w, h] → XYXY [x1, y1, x2, y2] に変換
    xyxy = boxes.clone()
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]

    # tv_tensors.BoundingBoxes でラップ (v2 が認識して同時変換)
    tv_boxes = tv_tensors.BoundingBoxes(
        xyxy, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
    )

    # augmentation 適用
    out_image, out_boxes, out_labels = augmentation(image, tv_boxes, labels)

    # 面積ゼロの bbox を除外
    if len(out_boxes) > 0:
        widths = out_boxes[:, 2] - out_boxes[:, 0]
        heights = out_boxes[:, 3] - out_boxes[:, 1]
        valid = (widths > 0) & (heights > 0)
        out_boxes = out_boxes[valid]
        out_labels = out_labels[valid]

    # XYXY → COCO [x, y, w, h] に戻す
    coco_boxes = torch.zeros_like(out_boxes)
    coco_boxes[:, 0] = out_boxes[:, 0]
    coco_boxes[:, 1] = out_boxes[:, 1]
    coco_boxes[:, 2] = out_boxes[:, 2] - out_boxes[:, 0]
    coco_boxes[:, 3] = out_boxes[:, 3] - out_boxes[:, 1]

    return out_image, coco_boxes, out_labels
