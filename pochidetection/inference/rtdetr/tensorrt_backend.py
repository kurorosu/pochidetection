"""TensorRT 推論バックエンド."""

import logging
from pathlib import Path
from typing import Any

import torch

try:
    import tensorrt as trt

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False

from pochidetection.inference.validation import validate_inputs, validate_model_file
from pochidetection.interfaces import IInferenceBackend
from pochidetection.logging import LoggerManager
from pochidetection.tensorrt.memory import TensorBinding, allocate_bindings

logger: logging.Logger = LoggerManager().get_logger(__name__)


class RTDetrTensorRTBackend(IInferenceBackend[tuple[torch.Tensor, torch.Tensor]]):
    """TensorRT エンジンを使用した推論バックエンド.

    PyTorch CUDA テンソルをバッファとして使用し,
    execute_async_v3 で非同期推論を実行する.

    Attributes:
        _engine: TensorRT エンジン.
        _context: TensorRT 実行コンテキスト.
        _stream: 推論用 CUDA ストリーム.
        _bindings: I/O テンソルバインディング.
    """

    def __init__(
        self,
        engine_path: Path | str,
    ) -> None:
        """初期化.

        Args:
            engine_path: TensorRT エンジンファイル (.engine) のパス.

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

        validate_model_file(engine_path, "TensorRTエンジン", ".engine")

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

        # 動的バッチサイズの入力テンソルに対して shape を確定させる.
        # DetectionPipeline はバッチサイズ1で画像を逐次処理するため,
        # バッチ次元を1に固定する.
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

        # 非デフォルト CUDA ストリームを使用.
        # デフォルトストリーム (stream 0) では execute_async_v3 内部で
        # 毎回 cudaStreamSynchronize() が追加呼び出しされ性能低下するため.
        self._stream = torch.cuda.Stream()

        logger.info(f"TensorRT backend loaded: {engine_path}")
        for b in self._bindings:
            kind = "input" if b.is_input else "output"
            logger.debug(f"  {kind}: {b.name}, shape={b.shape}, dtype={b.numpy_dtype}")

    def infer(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """推論を実行する.

        入力テンソルを GPU バッファにコピーし, TensorRT エンジンで推論後,
        出力を torch.Tensor として返す.

        Args:
            inputs: 前処理済みの入力テンソル辞書.
                キー "pixel_values" に (B, C, H, W) のテンソルを含む.

        Returns:
            pred_logits と pred_boxes のタプル.

        Raises:
            ValueError: 必須入力が不足している場合.
        """
        validate_inputs(inputs, self._input_names, "TensorRT")

        # 入力テンソルを GPU バッファにコピー
        self._stream.wait_stream(torch.cuda.default_stream())
        for binding in self._input_bindings:
            src = inputs[binding.name]
            if not src.is_cuda:
                src = src.cuda()
            with torch.cuda.stream(self._stream):
                binding.device_tensor.copy_(src)

        # 推論実行
        self._context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        # 出力テンソルを名前ベースで取得
        pred_logits = self._resolve_output(("logits", "pred_logits")).clone()
        pred_boxes = self._resolve_output(("pred_boxes",)).clone()
        return pred_logits, pred_boxes

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
            f"RT-DETR は 'logits'/'pred_logits' と 'pred_boxes' の出力が必要です."
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
