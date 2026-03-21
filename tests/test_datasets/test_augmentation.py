"""Data Augmentation パイプラインのテスト."""

import torch
from PIL import Image
from torchvision.transforms import v2

from pochidetection.configs.schemas import AugmentationConfig
from pochidetection.datasets.augmentation import apply_augmentation, build_augmentation


class TestBuildAugmentation:
    """build_augmentation のテスト."""

    def test_returns_compose_when_enabled(self) -> None:
        """enabled=True で transforms ありの場合に v2.Compose を返す."""
        config = AugmentationConfig.model_validate(
            {
                "enabled": True,
                "transforms": [{"name": "RandomHorizontalFlip", "p": 0.5}],
            }
        )
        result = build_augmentation(config)
        assert isinstance(result, v2.Compose)

    def test_returns_none_when_disabled(self) -> None:
        """enabled=False の場合に None を返す."""
        config = AugmentationConfig.model_validate(
            {
                "enabled": False,
                "transforms": [{"name": "RandomHorizontalFlip", "p": 0.5}],
            }
        )
        assert build_augmentation(config) is None

    def test_returns_none_when_empty_transforms(self) -> None:
        """transforms が空の場合に None を返す."""
        config = AugmentationConfig.model_validate({"enabled": True, "transforms": []})
        assert build_augmentation(config) is None

    def test_skips_unknown_transform(self) -> None:
        """不明な変換名をスキップして警告する."""
        config = AugmentationConfig.model_validate(
            {
                "enabled": True,
                "transforms": [
                    {"name": "NonExistentTransform"},
                    {"name": "RandomHorizontalFlip", "p": 0.5},
                ],
            }
        )
        result = build_augmentation(config)
        assert isinstance(result, v2.Compose)

    def test_color_jitter_with_params(self) -> None:
        """ColorJitter に追加パラメータが渡されることを確認."""
        config = AugmentationConfig.model_validate(
            {
                "enabled": True,
                "transforms": [
                    {
                        "name": "ColorJitter",
                        "brightness": 0.2,
                        "contrast": 0.3,
                    },
                ],
            }
        )
        result = build_augmentation(config)
        assert isinstance(result, v2.Compose)

    def test_random_apply_wrapping(self) -> None:
        """p < 1.0 の色変換が RandomApply でラップされることを確認."""
        config = AugmentationConfig.model_validate(
            {
                "enabled": True,
                "transforms": [{"name": "ColorJitter", "brightness": 0.2, "p": 0.3}],
            }
        )
        result = build_augmentation(config)
        assert isinstance(result, v2.Compose)
        # 内部の変換が RandomApply でラップされている
        assert isinstance(result.transforms[0], v2.RandomApply)


class TestApplyAugmentation:
    """apply_augmentation のテスト."""

    def _create_image(self) -> Image.Image:
        """100x100 のテスト用 RGB 画像を生成."""
        return Image.new("RGB", (100, 100), color=(128, 64, 32))

    def test_identity_transform(self) -> None:
        """無変換 (HorizontalFlip p=0) で bbox が維持されることを確認."""
        aug = v2.Compose([v2.RandomHorizontalFlip(p=0.0)])
        image = self._create_image()
        boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # [x, y, w, h]
        labels = torch.tensor([0])

        out_image, out_boxes, out_labels = apply_augmentation(aug, image, boxes, labels)

        assert out_image.size == image.size
        assert out_boxes.shape == (1, 4)
        assert out_labels.shape == (1,)
        # p=0 なので bbox は変わらない
        torch.testing.assert_close(out_boxes, boxes)

    def test_horizontal_flip_preserves_box_count(self) -> None:
        """水平反転後も bbox 数が維持されることを確認."""
        aug = v2.Compose([v2.RandomHorizontalFlip(p=1.0)])
        image = self._create_image()
        boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 10.0, 20.0, 30.0]])
        labels = torch.tensor([0, 1])

        out_image, out_boxes, out_labels = apply_augmentation(aug, image, boxes, labels)

        assert out_boxes.shape[0] == 2
        assert out_labels.shape[0] == 2

    def test_output_boxes_coco_format(self) -> None:
        """出力が COCO 形式 [x, y, w, h] であることを確認."""
        aug = v2.Compose([v2.RandomHorizontalFlip(p=1.0)])
        image = self._create_image()
        boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        labels = torch.tensor([0])

        _, out_boxes, _ = apply_augmentation(aug, image, boxes, labels)

        # w, h が正の値であること (COCO 形式)
        assert (out_boxes[:, 2] > 0).all()
        assert (out_boxes[:, 3] > 0).all()

    def test_empty_boxes(self) -> None:
        """空の bbox で apply_augmentation がエラーにならないことを確認."""
        aug = v2.Compose([v2.RandomHorizontalFlip(p=1.0)])
        image = self._create_image()
        boxes = torch.zeros((0, 4))
        labels = torch.zeros((0,), dtype=torch.int64)

        out_image, out_boxes, out_labels = apply_augmentation(aug, image, boxes, labels)

        assert out_boxes.shape == (0, 4)
        assert out_labels.shape == (0,)
