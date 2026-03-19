"""推論バックエンドの抽象インターフェース."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch

TOutput = TypeVar("TOutput")


class IInferenceBackend(ABC, Generic[TOutput]):
    """各種推論バックエンド (PyTorch, ONNX 等) の共通インターフェース.

    E2E 推論およびベンチマークパイプラインにおいて,
    バックエンドの差異を吸収するための Strategy として機能する.

    Type Parameters:
        TOutput: 推論出力の型.
            RT-DETR の場合: tuple[Tensor, Tensor] (pred_logits, pred_boxes).
            SSDLite の場合: dict[str, Tensor] (boxes, scores, labels).
    """

    @abstractmethod
    def infer(self, inputs: dict[str, torch.Tensor]) -> TOutput:
        """推論を実行する.

        Args:
            inputs: 前処理済みの入力テンソル辞書.
                キー "pixel_values" に (B, C, H, W) のテンソルを含む.

        Returns:
            モデル出力. 型はバックエンドの実装に依存する.
        """
        pass
