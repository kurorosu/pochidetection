"""DetectionCollatorのテスト."""

import pytest
import torch

from pochidetection.core import DetectionCollator


class TestDetectionCollator:
    """DetectionCollatorのテスト."""

    @pytest.fixture
    def sample_batch(self) -> list[dict[str, torch.Tensor | dict[str, torch.Tensor]]]:
        """サンプルバッチ (CocoDetectionDataset出力形式)."""
        return [
            {
                "pixel_values": torch.randn(3, 640, 640),
                "labels": {
                    "boxes": torch.tensor([[0.3, 0.3, 0.4, 0.4]]),
                    "class_labels": torch.tensor([0]),
                },
            },
            {
                "pixel_values": torch.randn(3, 640, 640),
                "labels": {
                    "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                    "class_labels": torch.tensor([1]),
                },
            },
        ]

    def test_collator_initialization(self) -> None:
        """Collatorの初期化を確認."""
        collator = DetectionCollator()
        assert collator is not None

    def test_collator_call(
        self,
        sample_batch: list[dict[str, torch.Tensor | dict[str, torch.Tensor]]],
    ) -> None:
        """Collatorの呼び出しを確認."""
        collator = DetectionCollator()
        result = collator(sample_batch)

        # 結果を確認
        assert "pixel_values" in result
        assert "labels" in result

        # pixel_valuesがバッチ化されていることを確認
        assert result["pixel_values"].shape == (2, 3, 640, 640)

        # labelsがリストであることを確認
        assert isinstance(result["labels"], list)
        assert len(result["labels"]) == 2

    def test_collator_labels_structure(
        self,
        sample_batch: list[dict[str, torch.Tensor | dict[str, torch.Tensor]]],
    ) -> None:
        """labelsの構造を確認."""
        collator = DetectionCollator()
        result = collator(sample_batch)

        for label in result["labels"]:
            assert "boxes" in label
            assert "class_labels" in label
            assert isinstance(label["boxes"], torch.Tensor)
            assert isinstance(label["class_labels"], torch.Tensor)

    def test_collator_empty_boxes(self) -> None:
        """空のボックスを処理できることを確認."""
        batch = [
            {
                "pixel_values": torch.randn(3, 640, 640),
                "labels": {
                    "boxes": torch.zeros((0, 4)),
                    "class_labels": torch.zeros((0,), dtype=torch.int64),
                },
            },
        ]

        collator = DetectionCollator()
        result = collator(batch)

        assert result["pixel_values"].shape == (1, 3, 640, 640)
        assert result["labels"][0]["boxes"].shape == (0, 4)
        assert result["labels"][0]["class_labels"].shape == (0,)

    def test_collator_single_sample(self) -> None:
        """単一サンプルを処理できることを確認."""
        batch = [
            {
                "pixel_values": torch.randn(3, 640, 640),
                "labels": {
                    "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
                    "class_labels": torch.tensor([2]),
                },
            },
        ]

        collator = DetectionCollator()
        result = collator(batch)

        assert result["pixel_values"].shape == (1, 3, 640, 640)
        assert len(result["labels"]) == 1
        torch.testing.assert_close(
            result["labels"][0]["boxes"],
            torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        )
        torch.testing.assert_close(
            result["labels"][0]["class_labels"],
            torch.tensor([2]),
        )
