"""IInferenceBackendのテスト."""

import numpy as np
import pytest
import torch

from pochidetection.interfaces import IInferenceBackend


class DummyBackend(IInferenceBackend[tuple[np.ndarray, np.ndarray]]):
    """テスト用のダミーバックエンド."""

    def infer(self, inputs: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        """ダミーの推論を実行."""
        logits = np.zeros((1, 5, 2))
        boxes = np.zeros((1, 5, 4))
        return logits, boxes


class TestIInferenceBackend:
    """IInferenceBackendの実装テスト."""

    def test_is_abstract_class(self) -> None:
        """直接インスタンス化できないことを確認."""
        with pytest.raises(TypeError):
            # 抽象クラスなのでインスタンス化不可
            _ = IInferenceBackend()  # type: ignore

    def test_has_required_methods(self) -> None:
        """必須メソッドが定義されていることを確認."""
        assert hasattr(IInferenceBackend, "infer")
        assert getattr(IInferenceBackend.infer, "__isabstractmethod__")

    def test_concrete_implementation(self) -> None:
        """具象クラスでメソッドが呼べることを確認."""
        backend = DummyBackend()
        inputs = {"pixel_values": torch.zeros(1, 3, 64, 64)}

        logits, boxes = backend.infer(inputs)
        assert logits.shape == (1, 5, 2)
        assert boxes.shape == (1, 5, 4)
