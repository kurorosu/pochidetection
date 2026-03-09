"""SSDLite PyTorch 推論バックエンド."""

from typing import Any

import torch

from pochidetection.interfaces import IInferenceBackend
from pochidetection.models import SSDLiteModel


class SSDLitePyTorchBackend(IInferenceBackend):
    """SSDLiteModel を使用した PyTorch 推論バックエンド.

    Attributes:
        _model: SSDLiteModel インスタンス.
    """

    def __init__(self, model: SSDLiteModel) -> None:
        """初期化.

        Args:
            model: 推論に使用するデバイス転送・評価モード設定済みのモデル.
        """
        self._model = model

    def infer(self, inputs: Any) -> dict[str, torch.Tensor]:
        """推論を実行する.

        Args:
            inputs: 前処理済みの入力テンソル辞書.
                キー "pixel_values" に (1, C, H, W) のテンソルを含む.

        Returns:
            検出結果の辞書 (boxes, scores, labels).
        """
        pixel_values = inputs["pixel_values"]
        with torch.no_grad():
            outputs = self._model(pixel_values)
        pred: dict[str, torch.Tensor] = outputs["predictions"][0]
        return pred

    def synchronize(self) -> None:
        """CUDA 同期.

        GPU を使用している場合のみ torch.cuda.synchronize() を呼び出す.
        """
        if torch.cuda.is_available():
            device = next(self._model.parameters()).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
