"""CocoDetectionDatasetのテスト."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import RTDetrImageProcessor

from pochidetection.datasets import CocoDetectionDataset
from pochidetection.interfaces import IDetectionDataset


class TestCocoDetectionDataset:
    """CocoDetectionDatasetのテスト."""

    @pytest.fixture
    def sample_dataset_dir(self, tmp_path: Path) -> Path:
        """サンプルデータセットを作成するfixture."""
        # 画像ディレクトリを作成
        images_dir = tmp_path / "JPEGImages"
        images_dir.mkdir()

        # ダミー画像を作成
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
            img.save(images_dir / f"image_{i}.jpg")

        # アノテーションを作成
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
    def processor(self) -> RTDetrImageProcessor:
        """RTDetrImageProcessorを作成するfixture."""
        return RTDetrImageProcessor()

    def test_implements_interface(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor
    ) -> None:
        """IDetectionDatasetを実装していることを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        assert isinstance(dataset, IDetectionDataset)

    def test_len(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor
    ) -> None:
        """__len__が正しい値を返すことを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        assert len(dataset) == 3

    def test_getitem_returns_correct_keys(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor
    ) -> None:
        """__getitem__が正しいキーを持つ辞書を返すことを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        sample = dataset[0]

        assert "pixel_values" in sample
        assert "labels" in sample
        assert "boxes" in sample["labels"]
        assert "class_labels" in sample["labels"]

    def test_getitem_boxes_format(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor
    ) -> None:
        """ボックスが正しい形式で返されることを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        sample = dataset[0]

        boxes = sample["labels"]["boxes"]
        assert isinstance(boxes, torch.Tensor)
        assert boxes.dtype == torch.float32
        assert boxes.shape == (2, 4)  # 2つのボックス

        # 正規化cxcywh形式で返されることを確認
        # 元: [10, 10, 30, 30] (xywh) -> cx=(10+15)/100=0.25, cy=(10+15)/100=0.25, w=0.3, h=0.3
        expected_first_box = torch.tensor([0.25, 0.25, 0.3, 0.3], dtype=torch.float32)
        assert torch.allclose(boxes[0], expected_first_box)

    def test_getitem_labels_format(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor
    ) -> None:
        """ラベルが正しい形式で返されることを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        sample = dataset[0]

        labels = sample["labels"]["class_labels"]
        assert isinstance(labels, torch.Tensor)
        assert labels.dtype == torch.int64
        assert labels.shape == (2,)  # 2つのラベル

        # カテゴリIDが連続インデックスに変換されていることを確認
        # category_id 1 -> index 0, category_id 2 -> index 1
        assert labels[0].item() == 0  # cat
        assert labels[1].item() == 1  # dog

    def test_getitem_no_annotations(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor
    ) -> None:
        """アノテーションがない画像でも正しく動作することを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        sample = dataset[2]  # image_2 にはアノテーションなし

        boxes = sample["labels"]["boxes"]
        labels = sample["labels"]["class_labels"]

        assert boxes.shape == (0, 4)
        assert labels.shape == (0,)

    def test_get_categories(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor
    ) -> None:
        """get_categoriesが正しいカテゴリ情報を返すことを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        categories = dataset.get_categories()

        assert len(categories) == 2
        assert categories[0]["name"] == "cat"
        assert categories[1]["name"] == "dog"

    def test_get_num_classes(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor
    ) -> None:
        """get_num_classesが正しいクラス数を返すことを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        assert dataset.get_num_classes() == 2

    def test_get_category_names(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor
    ) -> None:
        """get_category_namesが正しいカテゴリ名を返すことを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        names = dataset.get_category_names()

        assert names == ["cat", "dog"]

    def test_annotation_file_not_found(
        self, tmp_path: Path, processor: RTDetrImageProcessor
    ) -> None:
        """アノテーションファイルが見つからない場合にエラーが発生することを確認."""
        with pytest.raises(FileNotFoundError):
            CocoDetectionDataset(tmp_path, processor)

    def test_custom_annotation_file(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor
    ) -> None:
        """カスタムアノテーションファイル名を指定できることを確認."""
        # annotations.json を instances_train.json にリネーム
        (sample_dataset_dir / "annotations.json").rename(
            sample_dataset_dir / "instances_train.json"
        )

        dataset = CocoDetectionDataset(
            sample_dataset_dir, processor, annotation_file="instances_train.json"
        )
        assert len(dataset) == 3

    def test_zero_size_bbox_is_skipped(self, tmp_path: Path) -> None:
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

        processor = RTDetrImageProcessor()
        dataset = CocoDetectionDataset(tmp_path, processor)
        sample = dataset[0]

        # w=0, h=0 の 2 件がスキップされ, 有効な 1 件のみ残る
        assert sample["labels"]["boxes"].shape == (1, 4)
        assert sample["labels"]["class_labels"].shape == (1,)
