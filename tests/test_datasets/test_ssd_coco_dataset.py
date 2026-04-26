"""SsdCocoDataset のテスト."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from pochidetection.configs.schemas import ImageSizeDict
from pochidetection.datasets import SsdCocoDataset
from pochidetection.interfaces import IDetectionDataset


class TestSsdCocoDataset:
    """SsdCocoDataset のテスト."""

    @pytest.fixture
    def sample_dataset_dir(self, tmp_path: Path) -> Path:
        """サンプルデータセットを作成する fixture."""
        images_dir = tmp_path / "JPEGImages"
        images_dir.mkdir()

        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
            img.save(images_dir / f"image_{i}.jpg")

        annotations = {
            "images": [
                {
                    "id": 0,
                    "file_name": "JPEGImages/image_0.jpg",
                    "width": 100,
                    "height": 100,
                },
                {
                    "id": 1,
                    "file_name": "JPEGImages/image_1.jpg",
                    "width": 100,
                    "height": 100,
                },
                {
                    "id": 2,
                    "file_name": "JPEGImages/image_2.jpg",
                    "width": 100,
                    "height": 100,
                },
            ],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": [10, 10, 30, 30],
                    "area": 900,
                    "iscrowd": 0,
                },
                {
                    "id": 1,
                    "image_id": 0,
                    "category_id": 2,
                    "bbox": [50, 50, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [20, 20, 40, 40],
                    "area": 1600,
                    "iscrowd": 0,
                },
                # image_2 にはアノテーションなし
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"},
            ],
        }

        with open(tmp_path / "annotations.json", "w") as f:
            json.dump(annotations, f)

        return tmp_path

    @pytest.fixture
    def image_size(self) -> ImageSizeDict:
        """テスト用の画像サイズ."""
        return {"height": 320, "width": 320}

    def test_implements_interface(
        self, sample_dataset_dir: Path, image_size: ImageSizeDict
    ) -> None:
        """IDetectionDataset を実装していることを確認."""
        dataset = SsdCocoDataset(sample_dataset_dir, image_size)
        assert isinstance(dataset, IDetectionDataset)

    def test_len(self, sample_dataset_dir: Path, image_size: ImageSizeDict) -> None:
        """__len__ が正しい値を返すことを確認."""
        dataset = SsdCocoDataset(sample_dataset_dir, image_size)
        assert len(dataset) == 3

    def test_getitem_returns_correct_keys(
        self, sample_dataset_dir: Path, image_size: ImageSizeDict
    ) -> None:
        """__getitem__ が正しいキーを持つ辞書を返すことを確認."""
        dataset = SsdCocoDataset(sample_dataset_dir, image_size)
        sample = dataset[0]

        assert "pixel_values" in sample
        assert "labels" in sample
        assert "boxes" in sample["labels"]
        assert "class_labels" in sample["labels"]

    def test_getitem_pixel_values_shape(
        self, sample_dataset_dir: Path, image_size: ImageSizeDict
    ) -> None:
        """pixel_values がリサイズ後のサイズであることを確認."""
        dataset = SsdCocoDataset(sample_dataset_dir, image_size)
        sample = dataset[0]

        assert sample["pixel_values"].shape == (3, 320, 320)

    def test_getitem_boxes_are_xyxy(
        self, sample_dataset_dir: Path, image_size: ImageSizeDict
    ) -> None:
        """ボックスが xyxy ピクセル座標であることを確認."""
        dataset = SsdCocoDataset(sample_dataset_dir, image_size)
        sample = dataset[0]

        boxes = sample["labels"]["boxes"]
        assert isinstance(boxes, torch.Tensor)
        assert boxes.dtype == torch.float32
        assert boxes.shape == (2, 4)

        # 元: [10, 10, 30, 30] (xywh), image 100x100 → 320x320
        # xyxy: [10, 10, 40, 40] → scaled: [32, 32, 128, 128]
        expected_first_box = torch.tensor([32.0, 32.0, 128.0, 128.0])
        assert torch.allclose(boxes[0], expected_first_box)

    def test_getitem_labels_are_0_indexed(
        self, sample_dataset_dir: Path, image_size: ImageSizeDict
    ) -> None:
        """ラベルが 0-indexed であることを確認."""
        dataset = SsdCocoDataset(sample_dataset_dir, image_size)
        sample = dataset[0]

        class_labels = sample["labels"]["class_labels"]
        assert isinstance(class_labels, torch.Tensor)
        assert class_labels.dtype == torch.int64
        assert class_labels.shape == (2,)

        # category_id 1 → idx 0, category_id 2 → idx 1
        assert class_labels[0].item() == 0  # cat
        assert class_labels[1].item() == 1  # dog

    def test_getitem_no_annotations(
        self, sample_dataset_dir: Path, image_size: ImageSizeDict
    ) -> None:
        """アノテーションがない画像でも正しく動作することを確認."""
        dataset = SsdCocoDataset(sample_dataset_dir, image_size)
        sample = dataset[2]

        boxes = sample["labels"]["boxes"]
        class_labels = sample["labels"]["class_labels"]

        assert boxes.shape == (0, 4)
        assert class_labels.shape == (0,)

    def test_pixel_values_are_zero_one_range(
        self, sample_dataset_dir: Path, image_size: ImageSizeDict
    ) -> None:
        """pixel_values が [0, 1] 範囲であることを確認.

        SSDLite は GeneralizedRCNNTransform が内部で正規化するため,
        Dataset は ImageNet 正規化を適用せず [0, 1] のまま返す.
        """
        dataset = SsdCocoDataset(sample_dataset_dir, image_size)
        sample = dataset[0]

        pixel_values = sample["pixel_values"]
        assert pixel_values.min() >= 0.0
        assert pixel_values.max() <= 1.0

    def test_annotation_file_not_found(
        self, tmp_path: Path, image_size: ImageSizeDict
    ) -> None:
        """アノテーションファイルが見つからない場合にエラーが発生することを確認."""
        with pytest.raises(FileNotFoundError):
            SsdCocoDataset(tmp_path, image_size)

    def test_letterbox_landscape_image_pads_bbox_vertically(
        self, tmp_path: Path, image_size: ImageSizeDict
    ) -> None:
        """横長画像 + letterbox=True で bbox が scale + pad_top 加算される.

        1280x720 画像を 320x320 にリサイズする場合, scale=0.25, new=320x180,
        pad_top=70 (上下 70 ずつ). 元 bbox [200, 100, 200, 200] (xywh) は
        xyxy に直すと [200, 100, 400, 300]. letterbox 後は
        [200*0.25, 100*0.25 + 70, 400*0.25, 300*0.25 + 70] = [50, 95, 100, 145].
        """
        images_dir = tmp_path / "JPEGImages"
        images_dir.mkdir()
        img = Image.new("RGB", (1280, 720), color=(100, 100, 100))
        img.save(images_dir / "landscape.jpg")

        annotations = {
            "images": [
                {
                    "id": 1,
                    "file_name": "JPEGImages/landscape.jpg",
                    "width": 1280,
                    "height": 720,
                },
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [200, 100, 200, 200],
                },
            ],
            "categories": [{"id": 1, "name": "cat"}],
        }
        with open(tmp_path / "annotations.json", "w") as f:
            json.dump(annotations, f)

        dataset = SsdCocoDataset(tmp_path, image_size, letterbox=True)
        sample = dataset[0]

        assert sample["pixel_values"].shape == (3, 320, 320)
        boxes = sample["labels"]["boxes"]
        assert boxes.shape == (1, 4)
        expected = torch.tensor([[50.0, 95.0, 100.0, 145.0]])
        torch.testing.assert_close(boxes, expected, atol=1.0, rtol=0.0)

    def test_letterbox_false_falls_back_to_simple_resize(
        self, tmp_path: Path, image_size: ImageSizeDict
    ) -> None:
        """letterbox=False で従来の v2.Resize (単純リサイズ) に戻る.

        1280x720 → 320x320 の単純リサイズは独立スケール (x: 0.25, y: 320/720).
        元 bbox [200, 100, 400, 300] (xyxy) → [50, 44.44, 100, 133.33].
        """
        images_dir = tmp_path / "JPEGImages"
        images_dir.mkdir()
        img = Image.new("RGB", (1280, 720), color=(100, 100, 100))
        img.save(images_dir / "landscape.jpg")

        annotations = {
            "images": [
                {
                    "id": 1,
                    "file_name": "JPEGImages/landscape.jpg",
                    "width": 1280,
                    "height": 720,
                },
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [200, 100, 200, 200],
                },
            ],
            "categories": [{"id": 1, "name": "cat"}],
        }
        with open(tmp_path / "annotations.json", "w") as f:
            json.dump(annotations, f)

        dataset = SsdCocoDataset(tmp_path, image_size, letterbox=False)
        sample = dataset[0]

        assert sample["pixel_values"].shape == (3, 320, 320)
        boxes = sample["labels"]["boxes"]
        scale_y = 320 / 720
        expected = torch.tensor([[50.0, 100.0 * scale_y, 100.0, 300.0 * scale_y]])
        # v2.Resize は非等比スケールの際に rounding が入るため ~0.5 pixel の誤差を許容.
        torch.testing.assert_close(boxes, expected, atol=1.0, rtol=0.0)

    def test_zero_size_bbox_is_skipped(
        self, tmp_path: Path, image_size: ImageSizeDict
    ) -> None:
        """w=0 または h=0 の bbox がスキップされることを確認."""
        images_dir = tmp_path / "JPEGImages"
        images_dir.mkdir()
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img.save(images_dir / "img.jpg")

        annotations = {
            "images": [
                {
                    "id": 1,
                    "file_name": "JPEGImages/img.jpg",
                    "width": 100,
                    "height": 100,
                },
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30]},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 0, 20]},
                {"id": 3, "image_id": 1, "category_id": 1, "bbox": [70, 70, 10, 0]},
            ],
            "categories": [{"id": 1, "name": "cat"}],
        }
        with open(tmp_path / "annotations.json", "w") as f:
            json.dump(annotations, f)

        dataset = SsdCocoDataset(tmp_path, image_size)
        sample = dataset[0]

        assert sample["labels"]["boxes"].shape == (1, 4)
        assert sample["labels"]["class_labels"].shape == (1,)
