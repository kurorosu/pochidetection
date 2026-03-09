"""SSDLite ONNX Runtime 推論バックエンド."""

import logging
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.ops import batched_nms

from pochidetection.interfaces import IInferenceBackend
from pochidetection.logging import LoggerManager

logger: logging.Logger = LoggerManager().get_logger(__name__)

# BoxCoder の重み (torchvision SSD デフォルト)
_BOX_CODER_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
# exp() オーバーフロー防止用クランプ値
_BBOX_XFORM_CLIP = math.log(1000.0 / 16)


class SSDLiteOnnxBackend(IInferenceBackend):
    """SSDLite ONNX Runtime 推論バックエンド.

    ONNX モデルの生出力 (cls_logits, bbox_regression) から
    後処理 (softmax, アンカーデコード, NMS) を行い,
    検出結果を dict[str, torch.Tensor] で返す.

    Attributes:
        _session: ONNX Runtime の InferenceSession.
        _anchors: アンカーボックス座標 (num_anchors, 4), xyxy ピクセル座標.
    """

    def __init__(
        self,
        model_path: Path,
        num_classes: int,
        image_size: tuple[int, int],
        nms_iou_threshold: float = 0.55,
        score_thresh: float = 0.001,
        topk_candidates: int = 300,
        detections_per_img: int = 300,
        providers: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        """初期化.

        Args:
            model_path: ONNX モデルファイルのパス.
            num_classes: クラス数 (背景クラスを含まない).
            image_size: 入力画像サイズ (height, width).
            nms_iou_threshold: NMS の IoU 閾値.
            score_thresh: pre-NMS のスコア閾値.
            topk_candidates: pre-NMS の上位候補数.
            detections_per_img: NMS 後の最大検出数.
            providers: Execution Providers のリスト.
                None の場合, device に応じて自動選択する.
            device: 推論デバイス ("cpu" または "cuda").

        Raises:
            FileNotFoundError: モデルファイルが存在しない場合.
            ValueError: model_path がファイルでない, または .onnx でない場合.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"ONNXモデルが見つかりません: {model_path}")
        if not model_path.is_file():
            raise ValueError(
                f"ONNXモデルのパスはファイルである必要があります: {model_path}"
            )
        if model_path.suffix.lower() != ".onnx":
            raise ValueError(
                f"ONNXモデルのファイル拡張子は .onnx である必要があります: {model_path}"
            )

        self._num_classes = num_classes
        self._image_size = image_size
        self._nms_iou_threshold = nms_iou_threshold
        self._score_thresh = score_thresh
        self._topk_candidates = topk_candidates
        self._detections_per_img = detections_per_img

        if providers is None:
            providers = self._resolve_providers(device)

        ort.set_default_logger_severity(3)
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_names = tuple(inp.name for inp in self._session.get_inputs())
        self._output_names = tuple(out.name for out in self._session.get_outputs())
        self._input_dtype = self._session.get_inputs()[0].type

        active_providers = self._session.get_providers()
        logger.info(f"ONNX Runtime providers: {active_providers}")

        self._anchors = self._generate_anchors()
        logger.info(
            f"アンカー生成完了: {self._anchors.shape[0]} boxes, "
            f"image_size={image_size}"
        )

    @staticmethod
    def _resolve_providers(device: str) -> list[str]:
        """デバイス設定に応じた Execution Providers を返す.

        Args:
            device: 推論デバイス ("cpu" または "cuda").

        Returns:
            Execution Providers のリスト.
        """
        if device == "cuda":
            available = ort.get_available_providers()
            providers: list[str] = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            return providers
        return ["CPUExecutionProvider"]

    def _generate_anchors(self) -> torch.Tensor:
        """アンカーボックスを動的生成する.

        軽量な SSD モデル (重みロード不要) を構築し,
        backbone のダミー forward で grid_sizes を取得.
        DefaultBoxGenerator でアンカーを生成し, xyxy ピクセル座標に変換する.

        Returns:
            アンカーボックス (num_anchors, 4), xyxy ピクセル座標.
        """
        h, w = self._image_size
        ssd_num_classes = self._num_classes + 1

        dummy_model = ssdlite320_mobilenet_v3_large(
            weights_backbone=None, num_classes=ssd_num_classes
        )
        dummy_model.eval()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, h, w)
            features = dummy_model.backbone(dummy_input)

        grid_sizes = [list(f.shape[-2:]) for f in features.values()]

        anchor_generator: DefaultBoxGenerator = dummy_model.anchor_generator
        dboxes = anchor_generator._grid_default_boxes(grid_sizes, self._image_size)

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

    def infer(self, inputs: Any) -> dict[str, torch.Tensor]:
        """推論を実行する.

        ONNX Runtime で推論し, 後処理 (softmax, デコード, NMS) を適用して
        検出結果を返す.

        Args:
            inputs: 前処理済みの入力テンソル辞書.
                キー "pixel_values" に (1, C, H, W) のテンソルを含む.

        Returns:
            検出結果の辞書 (boxes, scores, labels).
        """
        missing = [name for name in self._input_names if name not in inputs]
        if missing:
            raise ValueError(
                f"ONNX入力が不足しています: {missing}. "
                f"利用可能なキー: {list(inputs.keys())}"
            )

        # FP16 モデルの場合は FP16 で入力, それ以外は FP32
        is_fp16 = self._input_dtype == "tensor(float16)"
        numpy_inputs: dict[str, np.ndarray] = {}
        for name in self._input_names:
            tensor = inputs[name].cpu()
            if is_fp16:
                numpy_inputs[name] = tensor.half().numpy()
            else:
                numpy_inputs[name] = tensor.float().numpy()

        raw_outputs = self._session.run(None, numpy_inputs)
        outputs_by_name = dict(zip(self._output_names, raw_outputs))

        # 後処理は常に FP32
        cls_logits = torch.from_numpy(outputs_by_name["cls_logits"].astype(np.float32))
        bbox_regression = torch.from_numpy(
            outputs_by_name["bbox_regression"].astype(np.float32)
        )

        # B=1 前提, batch 次元を除去
        cls_logits = cls_logits[0]  # (num_anchors, num_classes+1)
        bbox_regression = bbox_regression[0]  # (num_anchors, 4)

        return self._postprocess(cls_logits, bbox_regression)

    def _postprocess(
        self,
        cls_logits: torch.Tensor,
        bbox_regression: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """後処理を実行する.

        torchvision SSD の postprocess_detections と等価な処理を行う.

        Args:
            cls_logits: クラスロジット (num_anchors, num_classes+1).
            bbox_regression: ボックス回帰値 (num_anchors, 4).

        Returns:
            検出結果の辞書 (boxes, scores, labels).
        """
        # softmax でクラス確率に変換
        scores_all = F.softmax(cls_logits, dim=-1)

        # ボックスデコード
        boxes = self._decode_boxes(bbox_regression, self._anchors)

        # 画像サイズでクリップ
        h, w = self._image_size
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h)

        # per-class 処理
        all_boxes: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        # label=1..num_classes (背景クラス 0 を除外)
        ssd_num_classes = self._num_classes + 1
        for class_idx in range(1, ssd_num_classes):
            class_scores = scores_all[:, class_idx]

            # スコア閾値でフィルタ
            mask = class_scores > self._score_thresh
            filtered_scores = class_scores[mask]
            filtered_boxes = boxes[mask]

            if filtered_scores.numel() == 0:
                continue

            # topk 候補に絞る
            if filtered_scores.numel() > self._topk_candidates:
                topk_scores, topk_indices = filtered_scores.topk(self._topk_candidates)
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
        keep = batched_nms(cat_boxes, cat_scores, cat_labels, self._nms_iou_threshold)

        # detections_per_img でキャップ
        keep = keep[: self._detections_per_img]

        return {
            "boxes": cat_boxes[keep],
            "scores": cat_scores[keep],
            "labels": cat_labels[keep],
        }

    @staticmethod
    def _decode_boxes(rel_codes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Boxcoder デコード.

        アンカー相対のオフセットを xyxy 絶対座標に変換する.
        torchvision の BoxCoder.decode_single と等価.

        Args:
            rel_codes: 回帰オフセット (N, 4).
            anchors: アンカーボックス (N, 4), xyxy 形式.

        Returns:
            デコード済みボックス (N, 4), xyxy 形式.
        """
        wx, wy, ww, wh = _BOX_CODER_WEIGHTS

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

    def synchronize(self) -> None:
        """同期処理. ONNX Runtime は同期実行のため何もしない."""

    @property
    def active_providers(self) -> list[str]:
        """実際に使用されている Execution Providers を取得."""
        return cast(list[str], self._session.get_providers())

    @property
    def session(self) -> ort.InferenceSession:
        """ONNX Runtime セッションを取得."""
        return self._session

    @property
    def anchors(self) -> torch.Tensor:
        """生成済みアンカーボックスを取得."""
        return self._anchors
