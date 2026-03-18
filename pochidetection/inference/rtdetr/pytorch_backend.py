"""PyTorch 推論バックエンド."""

import torch

from pochidetection.inference.sync import synchronize_cuda
from pochidetection.interfaces import IInferenceBackend
from pochidetection.models import RTDetrModel


class RTDetrPyTorchBackend(IInferenceBackend[tuple[torch.Tensor, torch.Tensor]]):
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
            inputs: 前処理済みの入力テンソル辞書.

        Returns:
            pred_logits と pred_boxes のタプル.
        """
        outputs = self._model.model(**inputs)
        synchronize_cuda(self._model)
        return outputs.logits, outputs.pred_boxes
