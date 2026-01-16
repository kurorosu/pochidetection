"""CocoDetectionDatasetのテスト."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

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

    def test_implements_interface(self, sample_dataset_dir: Path) -> None:
        """IDetectionDatasetを実装していることを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir)
        assert isinstance(dataset, IDetectionDataset)

    def test_len(self, sample_dataset_dir: Path) -> None:
        """__len__が正しい値を返すことを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir)
        assert len(dataset) == 3

    def test_getitem_returns_correct_keys(self, sample_dataset_dir: Path) -> None:
        """__getitem__が正しいキーを持つ辞書を返すことを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir)
        sample = dataset[0]

        assert "image" in sample
        assert "boxes" in sample
        assert "labels" in sample
        assert "image_id" in sample
        assert "orig_size" in sample

    def test_getitem_boxes_format(self, sample_dataset_dir: Path) -> None:
        """ボックスが正しい形式で返されることを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir)
        sample = dataset[0]

        boxes = sample["boxes"]
        assert isinstance(boxes, torch.Tensor)
        assert boxes.dtype == torch.float32
        assert boxes.shape == (2, 4)  # 2つのボックス

        # COCO形式 [x, y, w, h] から [x_min, y_min, x_max, y_max] に変換されていることを確認
        # 元: [10, 10, 30, 30] -> [10, 10, 40, 40]
        expected_first_box = torch.tensor([10, 10, 40, 40], dtype=torch.float32)
        assert torch.allclose(boxes[0], expected_first_box)

    def test_getitem_labels_format(self, sample_dataset_dir: Path) -> None:
        """ラベルが正しい形式で返されることを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir)
        sample = dataset[0]

        labels = sample["labels"]
        assert isinstance(labels, torch.Tensor)
        assert labels.dtype == torch.int64
        assert labels.shape == (2,)  # 2つのラベル

        # カテゴリIDが連続インデックスに変換されていることを確認
        # category_id 1 -> index 0, category_id 2 -> index 1
        assert labels[0].item() == 0  # cat
        assert labels[1].item() == 1  # dog

    def test_getitem_no_annotations(self, sample_dataset_dir: Path) -> None:
        """アノテーションがない画像でも正しく動作することを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir)
        sample = dataset[2]  # image_2 にはアノテーションなし

        boxes = sample["boxes"]
        labels = sample["labels"]

        assert boxes.shape == (0, 4)
        assert labels.shape == (0,)

    def test_get_categories(self, sample_dataset_dir: Path) -> None:
        """get_categoriesが正しいカテゴリ情報を返すことを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir)
        categories = dataset.get_categories()

        assert len(categories) == 2
        assert categories[0]["name"] == "cat"
        assert categories[1]["name"] == "dog"

    def test_get_num_classes(self, sample_dataset_dir: Path) -> None:
        """get_num_classesが正しいクラス数を返すことを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir)
        assert dataset.get_num_classes() == 2

    def test_get_category_names(self, sample_dataset_dir: Path) -> None:
        """get_category_namesが正しいカテゴリ名を返すことを確認."""
        dataset = CocoDetectionDataset(sample_dataset_dir)
        names = dataset.get_category_names()

        assert names == ["cat", "dog"]

    def test_annotation_file_not_found(self, tmp_path: Path) -> None:
        """アノテーションファイルが見つからない場合にエラーが発生することを確認."""
        with pytest.raises(FileNotFoundError):
            CocoDetectionDataset(tmp_path)

    def test_custom_annotation_file(self, sample_dataset_dir: Path) -> None:
        """カスタムアノテーションファイル名を指定できることを確認."""
        # annotations.json を instances_train.json にリネーム
        (sample_dataset_dir / "annotations.json").rename(
            sample_dataset_dir / "instances_train.json"
        )

        dataset = CocoDetectionDataset(
            sample_dataset_dir, annotation_file="instances_train.json"
        )
        assert len(dataset) == 3
