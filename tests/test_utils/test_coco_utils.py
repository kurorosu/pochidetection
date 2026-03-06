"""coco_utils のテスト."""

import json
from pathlib import Path

from pochidetection.utils.coco_utils import (
    CocoGroundTruth,
    extract_basename,
    load_coco_ground_truth,
    xywh_to_xyxy,
)


def _create_coco_json(tmp_path: Path, data: dict) -> Path:
    """テスト用 COCO JSON を書き出す."""
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestExtractBasename:
    """extract_basename のテスト."""

    def test_backslash_path(self) -> None:
        """バックスラッシュ付きパスからベースネームを抽出."""
        assert extract_basename("JPEGImages\\img.jpg") == "img.jpg"

    def test_slash_path(self) -> None:
        """スラッシュ付きパスからベースネームを抽出."""
        assert extract_basename("JPEGImages/img.jpg") == "img.jpg"

    def test_plain_filename(self) -> None:
        """パス区切りなしのファイル名はそのまま返す."""
        assert extract_basename("img.jpg") == "img.jpg"

    def test_nested_path(self) -> None:
        """ネストされたパスからベースネームを抽出."""
        assert extract_basename("a/b/c/img.jpg") == "img.jpg"


class TestXywhToXyxy:
    """xywh_to_xyxy の変換テスト."""

    def test_conversion(self) -> None:
        """[x, y, w, h] -> [x1, y1, x2, y2] の変換を確認."""
        assert xywh_to_xyxy([10.0, 20.0, 50.0, 60.0]) == [10.0, 20.0, 60.0, 80.0]

    def test_zero_size(self) -> None:
        """サイズ 0 のボックスの変換を確認."""
        assert xywh_to_xyxy([5.0, 5.0, 0.0, 0.0]) == [5.0, 5.0, 5.0, 5.0]


class TestLoadCocoGroundTruth:
    """load_coco_ground_truth のテスト."""

    def test_basic_loading(self, tmp_path: Path) -> None:
        """基本的な読み込みで全フィールドが構築されることを確認."""
        path = _create_coco_json(
            tmp_path,
            {
                "images": [
                    {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
                    {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 20, 30, 40],
                    },
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 2,
                        "bbox": [50, 60, 70, 80],
                    },
                ],
                "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
            },
        )

        gt = load_coco_ground_truth(path)

        assert isinstance(gt, CocoGroundTruth)
        assert gt.image_id_by_filename["img1.jpg"] == 1
        assert gt.image_id_by_filename["img2.jpg"] == 2
        assert len(gt.filenames_by_image_id) == 2
        assert gt.category_id_to_idx == {1: 0, 2: 1}
        assert len(gt.categories) == 2
        assert 1 in gt.gt_by_image_id
        assert 2 in gt.gt_by_image_id

    def test_basename_registration(self, tmp_path: Path) -> None:
        """サブディレクトリ付き file_name のベースネームが登録されることを確認."""
        path = _create_coco_json(
            tmp_path,
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "train/img.jpg",
                        "width": 640,
                        "height": 480,
                    },
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 20, 30, 40],
                    },
                ],
                "categories": [{"id": 1, "name": "cat"}],
            },
        )

        gt = load_coco_ground_truth(path)

        assert gt.image_id_by_filename["train/img.jpg"] == 1
        assert gt.image_id_by_filename["img.jpg"] == 1
        assert "img.jpg" in gt.filenames_by_image_id[1]

    def test_background_category_excluded(self, tmp_path: Path) -> None:
        """背景カテゴリがフィルタされ, そのアノテーションが除外されることを確認."""
        path = _create_coco_json(
            tmp_path,
            {
                "images": [
                    {"id": 1, "file_name": "img.jpg", "width": 640, "height": 480},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 0,
                        "bbox": [0, 0, 10, 10],
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 20, 30, 40],
                    },
                ],
                "categories": [
                    {"id": 0, "name": "background"},
                    {"id": 1, "name": "cat"},
                ],
            },
        )

        gt = load_coco_ground_truth(path)

        assert 0 not in gt.category_id_to_idx
        assert 1 in gt.category_id_to_idx
        # image_id=1 には cat のアノテーションのみ
        assert len(gt.gt_by_image_id[1]) == 1
        assert gt.gt_by_image_id[1][0]["category_id"] == 1


class TestCocoGroundTruthGtByFilename:
    """CocoGroundTruth.gt_by_filename のテスト."""

    def test_returns_gt_keyed_by_filename(self, tmp_path: Path) -> None:
        """ファイル名をキーとした GT 辞書が正しく構築されることを確認."""
        path = _create_coco_json(
            tmp_path,
            {
                "images": [
                    {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
                    {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 20, 30, 40],
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [50, 60, 70, 80],
                    },
                    {
                        "id": 3,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                    },
                ],
                "categories": [{"id": 1, "name": "cat"}],
            },
        )

        gt = load_coco_ground_truth(path)
        by_filename = gt.gt_by_filename()

        assert "img1.jpg" in by_filename
        assert "img2.jpg" in by_filename
        assert len(by_filename["img1.jpg"]) == 2
        assert len(by_filename["img2.jpg"]) == 1

    def test_basename_preferred_as_key(self, tmp_path: Path) -> None:
        """サブディレクトリ付きの場合, basename がキーとして使われることを確認."""
        path = _create_coco_json(
            tmp_path,
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "subdir/img.jpg",
                        "width": 640,
                        "height": 480,
                    },
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 20, 30, 40],
                    },
                ],
                "categories": [{"id": 1, "name": "cat"}],
            },
        )

        gt = load_coco_ground_truth(path)
        by_filename = gt.gt_by_filename()

        # basename ("img.jpg") が最短なのでキーになる
        assert "img.jpg" in by_filename
        assert len(by_filename["img.jpg"]) == 1
