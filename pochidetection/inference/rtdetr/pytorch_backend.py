"""PyTorch 推論バックエンド."""

from typing import Any

import torch

from pochidetection.interfaces import IInferenceBackend
from pochidetection.models import RTDetrModel


class RTDetrPyTorchBackend(IInferenceBackend):
    """PyTorch および RTDetrModel を使用した推論バックエンド."""

    def __init__(self, model: RTDetrModel) -> None:
        """初期化.

        Args:
            model: 推論に使用するデバイス転送・評価モード設定済みのモデル.
        """
        self._model = model

    def infer(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """推論を実行する.

        Args:
            inputs: モデルの forward に渡す引数辞書.

        Returns:
            pred_logits と pred_boxes のタプル.
        """
        outputs = self._model.model(**inputs)
        return outputs.logits, outputs.pred_boxes

    def synchronize(self) -> None:
        """CUDA同期.

        GPUを使用している場合のみ torch.cuda.synchronize() を呼び出す.
        """
        if torch.cuda.is_available():
            # デバイスから使用中かを確認して同期
            device = next(self._model.parameters()).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
