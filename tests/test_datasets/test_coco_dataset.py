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

    def test_debug_save_without_augmentation(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor, tmp_path: Path
    ) -> None:
        """augmentation 無効でも debug_save_count > 0 で画像が保存される.

        Issue #563: letterbox 等の preprocess 追加を見越して, augmentation の
        有無に関わらず学習画像のデバッグ保存を発火させる.
        """
        save_dir = tmp_path / "debug_out"
        save_dir.mkdir()

        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        dataset.debug_save_count = 5
        dataset.debug_save_dir = save_dir

        # annotations 付きの image_0 / image_1 を引く.
        dataset[0]
        dataset[1]

        saved = sorted(save_dir.glob("train_*.jpg"))
        assert len(saved) == 2
        assert saved[0].name == "train_0000.jpg"
        assert saved[1].name == "train_0001.jpg"

    def test_debug_save_caps_at_count(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor, tmp_path: Path
    ) -> None:
        """debug_save_count を超えて保存しない."""
        save_dir = tmp_path / "debug_out"
        save_dir.mkdir()

        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        dataset.debug_save_count = 1
        dataset.debug_save_dir = save_dir

        dataset[0]
        dataset[1]

        saved = list(save_dir.glob("train_*.jpg"))
        assert len(saved) == 1

    def test_debug_save_skipped_when_no_annotations(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor, tmp_path: Path
    ) -> None:
        """annotations が空の image は debug save 対象外 (カウントも進まない)."""
        save_dir = tmp_path / "debug_out"
        save_dir.mkdir()

        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        dataset.debug_save_count = 5
        dataset.debug_save_dir = save_dir

        # image_2 はアノテーション無し → skip, count 進まず
        dataset[2]
        assert list(save_dir.glob("train_*.jpg")) == []
        assert dataset.debug_saved == 0

        # 続けて annotations 付きを引けば 0000 から保存される
        dataset[0]
        saved = list(save_dir.glob("train_*.jpg"))
        assert [p.name for p in saved] == ["train_0000.jpg"]

    def test_debug_save_with_augmentation_reflects_augmented_image(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor, tmp_path: Path
    ) -> None:
        """augmentation あり経路でも debug save が発火し, 保存画像が反映される.

        HorizontalFlip (p=1.0) を適用し, 保存された bbox 座標が反転されていることで
        「augmentation 後の annotations を経由して保存されている」ことを確認する.
        """
        from pochidetection.configs.schemas import AugmentationConfig
        from pochidetection.datasets.augmentation import build_augmentation

        save_dir = tmp_path / "debug_out"
        save_dir.mkdir()

        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        dataset._augmentation = build_augmentation(
            AugmentationConfig.model_validate(
                {
                    "enabled": True,
                    "transforms": [{"name": "RandomHorizontalFlip", "p": 1.0}],
                }
            )
        )
        dataset.debug_save_count = 5
        dataset.debug_save_dir = save_dir

        # image_0 の元 bbox: [10, 10, 30, 30] (xywh) → flip 後は x = 100 - 10 - 30 = 60.
        dataset[0]

        saved = list(save_dir.glob("train_*.jpg"))
        assert len(saved) == 1
        # 返却された sample が flip 後であることも併せて確認 (debug save 経路が
        # augmentation 適用ステップを通っている証左).
        sample = dataset[0]
        first_box = sample["labels"]["boxes"][0]
        # cxcywh 正規化済み. x_center = (60 + 15) / 100 = 0.75.
        assert first_box[0].item() == pytest.approx(0.75, abs=1e-3)

    def test_debug_save_disabled_when_count_is_zero(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor, tmp_path: Path
    ) -> None:
        """debug_save_count=0 なら dir が設定されていても保存しない."""
        save_dir = tmp_path / "debug_out"
        save_dir.mkdir()

        dataset = CocoDetectionDataset(sample_dataset_dir, processor)
        dataset.debug_save_count = 0
        dataset.debug_save_dir = save_dir

        dataset[0]
        dataset[1]

        assert list(save_dir.glob("train_*.jpg")) == []

    def test_letterbox_landscape_image_produces_target_size_and_adjusted_bbox(
        self, tmp_path: Path, processor: RTDetrImageProcessor
    ) -> None:
        """letterbox=True で pixel_values は target サイズ, bbox (cxcywh) は
        letterbox 後座標を target 基準で正規化した値になる.

        1280x720 → (320, 320) (image_size 引数で明示):
        scale = 0.25, new = (180, 320), pad_top = 70.
        元 bbox [200, 100, 200, 200] (xywh) → letterbox 後 pixel xywh:
        [50, 95, 50, 50]. target_w=target_h=320 で正規化:
        cx=(50+25)/320=0.234, cy=(95+25)/320=0.375, nw=50/320=0.156, nh=50/320=0.156.
        """
        images_dir = tmp_path / "JPEGImages"
        images_dir.mkdir()
        img = Image.new("RGB", (1280, 720), color=(80, 80, 80))
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

        dataset = CocoDetectionDataset(
            tmp_path, processor, letterbox=True, image_size=(320, 320)
        )
        sample = dataset[0]

        assert sample["pixel_values"].shape == (3, 320, 320)
        boxes = sample["labels"]["boxes"]
        assert boxes.shape == (1, 4)
        expected_cxcywh = torch.tensor(
            [
                [
                    (50.0 + 25.0) / 320.0,
                    (95.0 + 25.0) / 320.0,
                    50.0 / 320.0,
                    50.0 / 320.0,
                ]
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(boxes, expected_cxcywh, atol=1e-3, rtol=0.0)

    def test_letterbox_false_uses_legacy_processor_path(
        self, sample_dataset_dir: Path, processor: RTDetrImageProcessor
    ) -> None:
        """letterbox=False で従来経路 (processor 内部 resize + 元画像基準 cxcywh).

        従来の挙動: 元 100x100 画像の bbox [10, 10, 30, 30] (xywh) を元画像基準で
        cxcywh 正規化 → cx=0.25, cy=0.25, w=0.3, h=0.3 (既存テストと同じ期待値).
        """
        dataset = CocoDetectionDataset(sample_dataset_dir, processor, letterbox=False)
        sample = dataset[0]

        boxes = sample["labels"]["boxes"]
        assert boxes.shape == (2, 4)
        expected_first = torch.tensor([0.25, 0.25, 0.3, 0.3], dtype=torch.float32)
        torch.testing.assert_close(boxes[0], expected_first)

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
