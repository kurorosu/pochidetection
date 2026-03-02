"""ONNX Runtime 推論バックエンド."""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
import torch

from pochidetection.interfaces import IInferenceBackend
from pochidetection.logging import LoggerManager

logger: logging.Logger = LoggerManager().get_logger(__name__)

SUPPORTED_ONNX_DTYPE = "tensor(float)"


class OnnxBackend(IInferenceBackend):
    """ONNX Runtime を使用した推論バックエンド.

    Attributes:
        _session: ONNX Runtime の InferenceSession.
    """

    def __init__(
        self,
        model_path: Path,
        providers: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        """初期化.

        Args:
            model_path: ONNX モデルファイルのパス.
            providers: Execution Providers のリスト.
                None の場合, device に応じて自動選択する.
            device: 推論デバイス ("cpu" または "cuda").
                providers が None の場合にのみ使用される.

        Raises:
            FileNotFoundError: モデルファイルが存在しない場合.
            ValueError: model_path がファイルでない, または .onnx でない場合.
            ValueError: モデルの入力 dtype が tensor(float) でない場合.
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

        if providers is None:
            providers = self._resolve_providers(device)

        # RT-DETR の ScatterND オペレータが CUDA EP で大量の WARNING を出すため,
        # ONNX Runtime の C++ ロガーを ERROR 以上に制限する.
        # NOTE: グローバル設定のため同一プロセス内の全セッションに影響する.
        ort.set_default_logger_severity(3)

        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_names = tuple(inp.name for inp in self._session.get_inputs())
        self._output_names = tuple(out.name for out in self._session.get_outputs())

        active_providers = self._session.get_providers()
        logger.info(f"ONNX Runtime providers: {active_providers}")

        self._validate_input_dtype()

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

    def _validate_input_dtype(self) -> None:
        """入力テンソルの dtype を検証する.

        Raises:
            ValueError: 入力 dtype が tensor(float) でない場合.
        """
        inputs = self._session.get_inputs()
        for inp in inputs:
            if inp.type != SUPPORTED_ONNX_DTYPE:
                raise ValueError(
                    f"非対応の入力dtype: {inp.name} の型は '{inp.type}' ですが, "
                    f"'{SUPPORTED_ONNX_DTYPE}' のみサポートしています"
                )

    def infer(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """推論を実行する.

        入力の torch.Tensor を numpy に変換して ONNX Runtime で推論し,
        結果を torch.Tensor に戻して返す.

        Args:
            inputs: 前処理済みの入力テンソル辞書.
                キー "pixel_values" に (B, C, H, W) のテンソルを含む.

        Returns:
            pred_logits と pred_boxes のタプル.
        """
        missing = [name for name in self._input_names if name not in inputs]
        if missing:
            raise ValueError(
                f"ONNX入力が不足しています: {missing}. "
                f"利用可能なキー: {list(inputs.keys())}"
            )

        numpy_inputs: dict[str, np.ndarray] = {
            name: inputs[name].cpu().float().numpy() for name in self._input_names
        }

        raw_outputs = self._session.run(None, numpy_inputs)
        outputs_by_name = dict(zip(self._output_names, raw_outputs))

        pred_logits = torch.from_numpy(
            self._resolve_output(outputs_by_name, ("logits", "pred_logits"))
        )
        pred_boxes = torch.from_numpy(
            self._resolve_output(outputs_by_name, ("pred_boxes",))
        )
        return pred_logits, pred_boxes

    @staticmethod
    def _resolve_output(
        outputs: dict[str, np.ndarray], candidates: tuple[str, ...]
    ) -> np.ndarray:
        """候補名から出力テンソルを解決する.

        Args:
            outputs: 出力名をキーとする辞書.
            candidates: 優先順の候補名タプル.

        Returns:
            マッチした出力テンソル.

        Raises:
            RuntimeError: どの候補名も見つからない場合.
        """
        for name in candidates:
            if name in outputs:
                return outputs[name]
        raise RuntimeError(
            f"ONNX出力に必要なテンソルが見つかりません. "
            f"候補: {candidates}, 利用可能: {list(outputs.keys())}"
        )

    def synchronize(self) -> None:
        """同期処理. ONNX Runtime は同期実行のため何もしない."""
        pass

    @property
    def active_providers(self) -> list[str]:
        """実際に使用されている Execution Providers を取得."""
        return cast(list[str], self._session.get_providers())

    @property
    def session(self) -> ort.InferenceSession:
        """ONNX Runtime セッションを取得."""
        return self._session
