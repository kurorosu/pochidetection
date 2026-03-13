"""SSDLite TensorRT 推論バックエンド."""

import logging
from pathlib import Path
from typing import Any

import torch

try:
    import tensorrt as trt

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False

from pochidetection.inference.ssdlite.postprocessing import (
    generate_anchors,
    postprocess,
)
from pochidetection.inference.validation import validate_inputs
from pochidetection.interfaces import IInferenceBackend
from pochidetection.logging import LoggerManager
from pochidetection.tensorrt.memory import TensorBinding, allocate_bindings

logger: logging.Logger = LoggerManager().get_logger(__name__)


class SSDLiteTensorRTBackend(IInferenceBackend):
    """TensorRT エンジンを使用した SSDLite 推論バックエンド.

    TensorRT エンジンの生出力 (cls_logits, bbox_regression) から
    後処理 (softmax, アンカーデコード, NMS) を行い,
    検出結果を dict[str, torch.Tensor] で返す.

    Attributes:
        _engine: TensorRT エンジン.
        _context: TensorRT 実行コンテキスト.
        _stream: 推論用 CUDA ストリーム.
        _bindings: I/O テンソルバインディング.
        _anchors: アンカーボックス座標 (num_anchors, 4), xyxy ピクセル座標.
    """

    def __init__(
        self,
        engine_path: Path | str,
        num_classes: int,
        image_size: tuple[int, int],
        nms_iou_threshold: float = 0.55,
        score_thresh: float = 0.001,
        topk_candidates: int = 300,
        detections_per_img: int = 300,
    ) -> None:
        """初期化.

        Args:
            engine_path: TensorRT エンジンファイル (.engine) のパス.
            num_classes: クラス数 (背景クラスを含まない).
            image_size: 入力画像サイズ (height, width).
            nms_iou_threshold: NMS の IoU 閾値.
            score_thresh: pre-NMS のスコア閾値.
            topk_candidates: pre-NMS の上位候補数.
            detections_per_img: NMS 後の最大検出数.

        Raises:
            ImportError: tensorrt がインストールされていない場合.
            FileNotFoundError: エンジンファイルが存在しない場合.
            ValueError: .engine 以外の拡張子が指定された場合.
            RuntimeError: エンジンのデシリアライズに失敗した場合.
        """
        if not _TRT_AVAILABLE:
            raise ImportError(
                "tensorrt パッケージがインストールされていません. "
                "GPU環境構築手順に従って TensorRT をインストールしてください."
            )

        engine_path = Path(engine_path)

        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRTエンジンが見つかりません: {engine_path}")
        if not engine_path.is_file():
            raise ValueError(
                f"TensorRTエンジンのパスはファイルである必要があります: {engine_path}"
            )
        if engine_path.suffix.lower() != ".engine":
            raise ValueError(
                f"TensorRTエンジンの拡張子は .engine である必要があります: "
                f"{engine_path}"
            )

        self._num_classes = num_classes
        self._image_size = image_size
        self._nms_iou_threshold = nms_iou_threshold
        self._score_thresh = score_thresh
        self._topk_candidates = topk_candidates
        self._detections_per_img = detections_per_img

        trt_logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self._engine = runtime.deserialize_cuda_engine(engine_data)
        if self._engine is None:
            raise RuntimeError(
                f"TensorRTエンジンのデシリアライズに失敗しました: {engine_path}"
            )

        self._context = self._engine.create_execution_context()

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                shape = list(self._engine.get_tensor_shape(name))
                if shape[0] == -1:
                    shape[0] = 1
                    self._context.set_input_shape(name, shape)

        self._bindings = allocate_bindings(self._engine, self._context)

        self._input_bindings = [b for b in self._bindings if b.is_input]
        self._output_bindings_by_name = {
            b.name: b for b in self._bindings if not b.is_input
        }
        self._input_names = tuple(b.name for b in self._input_bindings)

        self._stream = torch.cuda.Stream()

        self._anchors = generate_anchors(num_classes, image_size)

        logger.info(f"TensorRT backend loaded: {engine_path}")
        for b in self._bindings:
            kind = "input" if b.is_input else "output"
            logger.debug(f"  {kind}: {b.name}, shape={b.shape}, dtype={b.numpy_dtype}")
        logger.info(
            f"アンカー生成完了: {self._anchors.shape[0]} boxes, "
            f"image_size={image_size}"
        )

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """推論を実行する.

        入力テンソルを GPU バッファにコピーし, TensorRT エンジンで推論後,
        後処理 (softmax, デコード, NMS) を適用して検出結果を返す.

        Args:
            inputs: 前処理済みの入力テンソル辞書.
                キー "pixel_values" に (B, C, H, W) のテンソルを含む.

        Returns:
            検出結果の辞書 (boxes, scores, labels).

        Raises:
            ValueError: 必須入力が不足している場合.
        """
        validate_inputs(inputs, self._input_names, "TensorRT")

        self._stream.wait_stream(torch.cuda.default_stream())
        for binding in self._input_bindings:
            src = inputs[binding.name]
            if not src.is_cuda:
                src = src.cuda()
            with torch.cuda.stream(self._stream):
                binding.device_tensor.copy_(src)

        self._context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        cls_logits = self._resolve_output(("cls_logits",)).clone()
        bbox_regression = self._resolve_output(("bbox_regression",)).clone()

        # B=1 前提, batch 次元を除去
        cls_logits = cls_logits[0]  # (num_anchors, num_classes+1)
        bbox_regression = bbox_regression[0]  # (num_anchors, 4)

        # 後処理は CPU + FP32 で実行
        cls_logits = cls_logits.float().cpu()
        bbox_regression = bbox_regression.float().cpu()

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

    def _resolve_output(self, candidates: tuple[str, ...]) -> torch.Tensor:
        """候補名から出力バインディングのテンソルを解決する.

        Args:
            candidates: 優先順の候補名タプル.

        Returns:
            マッチした出力テンソル.

        Raises:
            RuntimeError: どの候補名も見つからない場合.
        """
        for name in candidates:
            if name in self._output_bindings_by_name:
                return self._output_bindings_by_name[name].device_tensor
        available = list(self._output_bindings_by_name.keys())
        raise RuntimeError(
            f"TensorRTエンジンに必要な出力テンソルが見つかりません. "
            f"候補: {candidates}, 利用可能な出力: {available}. "
            f"SSDLite は 'cls_logits' と 'bbox_regression' の出力が必要です."
        )

    def synchronize(self) -> None:
        """CUDA 同期. ストリームの完了を待機する."""
        self._stream.synchronize()

    @property
    def engine(self) -> Any:
        """Tensorrt エンジンを取得."""
        return self._engine

    @property
    def bindings(self) -> list[TensorBinding]:
        """I/O テンソルバインディングを取得."""
        return self._bindings
