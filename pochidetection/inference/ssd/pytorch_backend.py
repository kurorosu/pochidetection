"""SSD 共通 PyTorch 推論バックエンド."""

import torch

from pochidetection.inference.sync import synchronize_cuda
from pochidetection.interfaces import IInferenceBackend
from pochidetection.interfaces.model import IDetectionModel


class SsdPyTorchBackend(IInferenceBackend[dict[str, torch.Tensor]]):
    """IDetectionModel を使用した SSD 共通 PyTorch 推論バックエンド.

    SSDLite と SSD300 の両方で使用できる.

    Attributes:
        _model: IDetectionModel インスタンス.
    """

    def __init__(self, model: IDetectionModel) -> None:
        """初期化.

        Args:
            model: 推論に使用するデバイス転送・評価モード設定済みのモデル.
        """
        self._model = model

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
        synchronize_cuda(self._model)
        return pred
