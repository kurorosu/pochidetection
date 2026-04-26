"""SSDLite 後処理共通モジュール.

ONNX / TensorRT バックエンドで共通のアンカー生成, ボックスデコード,
NMS 後処理ロジックを提供する.
"""

import math

import torch
import torch.nn.functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.ops import batched_nms

# BoxCoder の重み (torchvision SSD デフォルト)
BOX_CODER_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
# exp() オーバーフロー防止用クランプ値
_BBOX_XFORM_CLIP = math.log(1000.0 / 16)


def generate_anchors(
    num_classes: int,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """アンカーボックスを動的生成する.

    軽量な SSD モデル (重みロード不要) を構築し,
    backbone のダミー forward で grid_sizes を取得.
    DefaultBoxGenerator でアンカーを生成し, xyxy ピクセル座標に変換する.

    Args:
        num_classes: クラス数 (背景クラスを含まない).
        image_size: 入力画像サイズ (height, width).

    Returns:
        アンカーボックス (num_anchors, 4), xyxy ピクセル座標.
    """
    h, w = image_size
    ssd_num_classes = num_classes + 1

    dummy_model = ssdlite320_mobilenet_v3_large(
        weights_backbone=None, num_classes=ssd_num_classes
    )
    dummy_model.eval()

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, h, w)
        features = dummy_model.backbone(dummy_input)

    grid_sizes = [list(f.shape[-2:]) for f in features.values()]

    anchor_generator: DefaultBoxGenerator = dummy_model.anchor_generator
    dboxes = anchor_generator._grid_default_boxes(grid_sizes, list(image_size))

    # cxcywh 正規化座標 → xyxy ピクセル座標
    x_y_size = torch.tensor([w, h], dtype=dboxes.dtype)
    anchors = torch.cat(
        [
            (dboxes[:, :2] - 0.5 * dboxes[:, 2:]) * x_y_size,
            (dboxes[:, :2] + 0.5 * dboxes[:, 2:]) * x_y_size,
        ],
        dim=-1,
    )
    return anchors


def decode_boxes(rel_codes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Boxcoder デコード.

    アンカー相対のオフセットを xyxy 絶対座標に変換する.
    torchvision の BoxCoder.decode_single と等価.

    Args:
        rel_codes: 回帰オフセット (N, 4).
        anchors: アンカーボックス (N, 4), xyxy 形式.

    Returns:
        デコード済みボックス (N, 4), xyxy 形式.
    """
    wx, wy, ww, wh = BOX_CODER_WEIGHTS

    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = rel_codes[:, 0] / wx
    dy = rel_codes[:, 1] / wy
    dw = rel_codes[:, 2] / ww
    dh = rel_codes[:, 3] / wh

    # exp() オーバーフロー防止
    dw = dw.clamp(max=_BBOX_XFORM_CLIP)
    dh = dh.clamp(max=_BBOX_XFORM_CLIP)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.stack(
        [
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ],
        dim=-1,
    )
    return pred_boxes


def postprocess(
    cls_logits: torch.Tensor,
    bbox_regression: torch.Tensor,
    anchors: torch.Tensor,
    num_classes: int,
    image_size: tuple[int, int],
    nms_iou_threshold: float,
    score_thresh: float = 0.001,
    topk_candidates: int = 300,
    detections_per_img: int = 300,
) -> dict[str, torch.Tensor]:
    """後処理を実行する.

    torchvision SSD の postprocess_detections と等価な処理を行う.

    Args:
        cls_logits: クラスロジット (num_anchors, num_classes+1).
        bbox_regression: ボックス回帰値 (num_anchors, 4).
        anchors: アンカーボックス (num_anchors, 4), xyxy ピクセル座標.
        num_classes: クラス数 (背景クラスを含まない).
        image_size: 入力画像サイズ (height, width).
        nms_iou_threshold: NMS の IoU 閾値.
        score_thresh: pre-NMS のスコア閾値.
        topk_candidates: pre-NMS の上位候補数.
        detections_per_img: NMS 後の最大検出数.

    Returns:
        検出結果の辞書 (boxes, scores, labels).
    """
    # softmax でクラス確率に変換
    scores_all = F.softmax(cls_logits, dim=-1)

    # ボックスデコード
    boxes = decode_boxes(bbox_regression, anchors)

    # 画像サイズでクリップ
    h, w = image_size
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h)

    # per-class 処理
    all_boxes: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    # label=1..num_classes (背景クラス 0 を除外)
    ssd_num_classes = num_classes + 1
    for class_idx in range(1, ssd_num_classes):
        class_scores = scores_all[:, class_idx]

        # スコア閾値でフィルタ
        mask = class_scores > score_thresh
        filtered_scores = class_scores[mask]
        filtered_boxes = boxes[mask]

        if filtered_scores.numel() == 0:
            continue

        # topk 候補に絞る
        if filtered_scores.numel() > topk_candidates:
            topk_scores, topk_indices = filtered_scores.topk(topk_candidates)
            filtered_scores = topk_scores
            filtered_boxes = filtered_boxes[topk_indices]

        all_boxes.append(filtered_boxes)
        all_scores.append(filtered_scores)
        # 0-indexed foreground ラベル
        all_labels.append(
            torch.full_like(filtered_scores, class_idx - 1, dtype=torch.int64)
        )

    # 候補が 0 件の場合
    if len(all_boxes) == 0:
        return {
            "boxes": torch.zeros(0, 4),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.int64),
        }

    # 全クラスの候補を結合
    cat_boxes = torch.cat(all_boxes, dim=0)
    cat_scores = torch.cat(all_scores, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)

    # NMS
    keep = batched_nms(cat_boxes, cat_scores, cat_labels, nms_iou_threshold)

    # detections_per_img でキャップ
    keep = keep[:detections_per_img]

    return {
        "boxes": cat_boxes[keep],
        "scores": cat_scores[keep],
        "labels": cat_labels[keep],
    }
