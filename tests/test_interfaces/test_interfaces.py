"""interfacesモジュールのテスト."""

from abc import ABC

import torch.nn as nn

from pochidetection.interfaces import (
    IDetectionDataset,
    IDetectionModel,
)


class TestIDetectionDataset:
    """IDetectionDatasetのテスト."""

    def test_is_abstract_class(self) -> None:
        """抽象クラスであることを確認."""
        assert issubclass(IDetectionDataset, ABC)

    def test_has_required_methods(self) -> None:
        """必須メソッドが定義されていることを確認."""
        assert hasattr(IDetectionDataset, "__len__")
        assert hasattr(IDetectionDataset, "__getitem__")
        assert hasattr(IDetectionDataset, "get_categories")


class TestIDetectionModel:
    """IDetectionModelのテスト."""

    def test_is_abstract_class(self) -> None:
        """抽象クラスであることを確認."""
        assert issubclass(IDetectionModel, ABC)

    def test_inherits_nn_module(self) -> None:
        """nn.Moduleを継承していることを確認."""
        assert issubclass(IDetectionModel, nn.Module)

    def test_has_required_methods(self) -> None:
        """必須メソッドが定義されていることを確認."""
        assert hasattr(IDetectionModel, "forward")
        assert hasattr(IDetectionModel, "get_backbone_params")
        assert hasattr(IDetectionModel, "get_head_params")
