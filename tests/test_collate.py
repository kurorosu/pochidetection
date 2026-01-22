"""DetectionCollatorのテスト."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from pochidetection.core import DetectionCollator


class TestDetectionCollator:
    """DetectionCollatorのテスト."""

    @pytest.fixture
    def mock_processor(self) -> MagicMock:
        """モックImageProcessor."""
        processor = MagicMock()
        processor.return_value = {"pixel_values": torch.randn(2, 3, 640, 640)}
        return processor

    @pytest.fixture
    def sample_batch(self) -> list[dict[str, torch.Tensor | Image.Image | int | tuple]]:
        """サンプルバッチ."""
        return [
            {
                "image": Image.new("RGB", (100, 100)),
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
                "image_id": 1,
                "orig_size": (100, 100),
            },
            {
                "image": Image.new("RGB", (200, 200)),
                "boxes": torch.tensor([[20.0, 20.0, 100.0, 100.0]]),
                "labels": torch.tensor([1]),
                "image_id": 2,
                "orig_size": (200, 200),
            },
        ]

    @patch("pochidetection.core.collate.RTDetrImageProcessor")
    def test_collator_initialization(self, mock_processor_class: MagicMock) -> None:
        """Collatorの初期化を確認."""
        collator = DetectionCollator(
            model_name="PekingU/rtdetr_r50vd",
            image_size=640,
        )
        mock_processor_class.from_pretrained.assert_called_once_with(
            "PekingU/rtdetr_r50vd"
        )

    @patch("pochidetection.core.collate.RTDetrImageProcessor")
    def test_collator_call(
        self,
        mock_processor_class: MagicMock,
        sample_batch: list[dict[str, torch.Tensor | Image.Image | int | tuple]],
    ) -> None:
        """Collatorの呼び出しを確認."""
        # モックの設定
        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.randn(2, 3, 640, 640)}
        mock_processor_class.from_pretrained.return_value = mock_processor

        collator = DetectionCollator()
        result = collator(sample_batch)

        # 結果を確認
        assert "pixel_values" in result
        assert "boxes" in result
        assert "labels" in result
        assert "image_ids" in result
        assert "orig_sizes" in result

        assert len(result["boxes"]) == 2
        assert len(result["labels"]) == 2
        assert result["image_ids"] == [1, 2]

    @patch("pochidetection.core.collate.RTDetrImageProcessor")
    def test_collator_box_normalization(
        self,
        mock_processor_class: MagicMock,
    ) -> None:
        """ボックスの正規化を確認."""
        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 640, 640)}
        mock_processor_class.from_pretrained.return_value = mock_processor

        # 100x100の画像で (10, 10, 50, 50) のボックス
        batch = [
            {
                "image": Image.new("RGB", (100, 100)),
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
                "image_id": 1,
                "orig_size": (100, 100),
            },
        ]

        collator = DetectionCollator()
        result = collator(batch)

        # 正規化されたボックス (cxcywh形式)
        # xyxy: (10, 10, 50, 50) / 100 = (0.1, 0.1, 0.5, 0.5)
        # cxcywh: cx=(0.1+0.5)/2=0.3, cy=(0.1+0.5)/2=0.3, w=0.5-0.1=0.4, h=0.5-0.1=0.4
        expected_box = torch.tensor([[0.3, 0.3, 0.4, 0.4]])
        torch.testing.assert_close(
            result["boxes"][0], expected_box, rtol=1e-4, atol=1e-4
        )

    @patch("pochidetection.core.collate.RTDetrImageProcessor")
    def test_collator_empty_boxes(
        self,
        mock_processor_class: MagicMock,
    ) -> None:
        """空のボックスを処理できることを確認."""
        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 640, 640)}
        mock_processor_class.from_pretrained.return_value = mock_processor

        batch = [
            {
                "image": Image.new("RGB", (100, 100)),
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": 1,
                "orig_size": (100, 100),
            },
        ]

        collator = DetectionCollator()
        result = collator(batch)

        assert result["boxes"][0].shape == (0, 4)
        assert result["labels"][0].shape == (0,)

    @patch("pochidetection.core.collate.RTDetrImageProcessor")
    def test_collator_tensor_image(
        self,
        mock_processor_class: MagicMock,
    ) -> None:
        """テンソル画像を処理できることを確認."""
        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 640, 640)}
        mock_processor_class.from_pretrained.return_value = mock_processor

        # (C, H, W) 形式のテンソル画像
        batch = [
            {
                "image": torch.rand(3, 100, 100),
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
                "image_id": 1,
                "orig_size": (100, 100),
            },
        ]

        collator = DetectionCollator()
        result = collator(batch)

        # 正常に処理されることを確認
        assert "pixel_values" in result
        assert len(result["boxes"]) == 1
