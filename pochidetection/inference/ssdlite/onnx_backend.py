"""SSDLite ONNX Runtime 推論バックエンド."""

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort
import torch

from pochidetection.inference.providers import resolve_providers
from pochidetection.inference.ssdlite.postprocessing import (
    generate_anchors,
    postprocess,
)
from pochidetection.interfaces import IInferenceBackend
from pochidetection.logging import LoggerManager

logger: logging.Logger = LoggerManager().get_logger(__name__)


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
            providers = resolve_providers(device)

        ort.set_default_logger_severity(3)
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_names = tuple(inp.name for inp in self._session.get_inputs())
        self._output_names = tuple(out.name for out in self._session.get_outputs())
        self._input_dtype = self._session.get_inputs()[0].type

        active_providers = self._session.get_providers()
        logger.info(f"ONNX Runtime providers: {active_providers}")

        self._anchors = generate_anchors(num_classes, image_size)
        logger.info(
            f"アンカー生成完了: {self._anchors.shape[0]} boxes, "
            f"image_size={image_size}"
        )

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

        return postprocess(
            cls_logits,
            bbox_regression,
            self._anchors,
            self._num_classes,
            self._image_size,
            self._nms_iou_threshold,
            self._score_thresh,
            self._topk_candidates,
            self._detections_per_img,
        )

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
