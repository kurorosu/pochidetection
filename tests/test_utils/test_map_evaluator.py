"""MapEvaluator のテスト."""

import json
from pathlib import Path

import pytest

from pochidetection.scripts.rtdetr.inference.detection import Detection
from pochidetection.utils.benchmark import DetectionMetrics
from pochidetection.utils.map_evaluator import MapEvaluator


def _create_coco_annotation(tmp_path: Path, images: list, annotations: list) -> Path:
    """テスト用 COCO アノテーション JSON を作成する."""
    data = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestMapEvaluatorPerfectMatch:
    """GT と完全一致する予測での mAP テスト."""

    def test_perfect_predictions_yield_map_one(self, tmp_path: Path) -> None:
        """GT と完全一致する予測で mAP が 1.0 になることを確認."""
        ann_path = _create_coco_annotation(
            tmp_path,
            images=[
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            ],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]},
            ],
        )

        evaluator = MapEvaluator(ann_path)
        predictions = {
            "img1.jpg": [
                Detection(
                    box=[10.0, 20.0, 60.0, 80.0],  # xywh -> xyxy: 10,20,60,80
                    score=0.99,
                    label=0,
                ),
            ],
        }

        result = evaluator.evaluate(predictions)

        assert isinstance(result, DetectionMetrics)
        assert result.map_50 == pytest.approx(1.0, abs=1e-3)
        assert result.map_50_95 == pytest.approx(1.0, abs=1e-3)


class TestMapEvaluatorNoMatch:
    """GT と全く一致しない予測での mAP テスト."""

    def test_wrong_predictions_yield_map_zero(self, tmp_path: Path) -> None:
        """GT と全く異なる予測で mAP が 0.0 になることを確認."""
        ann_path = _create_coco_annotation(
            tmp_path,
            images=[
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            ],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]},
            ],
        )

        evaluator = MapEvaluator(ann_path)
        predictions = {
            "img1.jpg": [
                Detection(
                    box=[500.0, 400.0, 600.0, 450.0],
                    score=0.99,
                    label=0,
                ),
            ],
        }

        result = evaluator.evaluate(predictions)

        assert result.map_50 == pytest.approx(0.0, abs=1e-3)
        assert result.map_50_95 == pytest.approx(0.0, abs=1e-3)


class TestMapEvaluatorEmptyPredictions:
    """予測が空の場合のテスト."""

    def test_no_predictions_yield_map_zero(self, tmp_path: Path) -> None:
        """予測が空のとき mAP が 0.0 になることを確認."""
        ann_path = _create_coco_annotation(
            tmp_path,
            images=[
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            ],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]},
            ],
        )

        evaluator = MapEvaluator(ann_path)
        predictions: dict[str, list[Detection]] = {"img1.jpg": []}

        result = evaluator.evaluate(predictions)

        assert result.map_50 == pytest.approx(0.0, abs=1e-3)
        assert result.map_50_95 == pytest.approx(0.0, abs=1e-3)


class TestMapEvaluatorUnknownFilename:
    """アノテーションにないファイル名が渡された場合のテスト."""

    def test_unknown_filename_is_skipped(self, tmp_path: Path) -> None:
        """アノテーションにないファイル名は無視されることを確認."""
        ann_path = _create_coco_annotation(
            tmp_path,
            images=[
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            ],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]},
            ],
        )

        evaluator = MapEvaluator(ann_path)
        predictions = {
            "unknown.jpg": [
                Detection(box=[10.0, 20.0, 60.0, 80.0], score=0.99, label=0),
            ],
            "img1.jpg": [
                Detection(box=[10.0, 20.0, 60.0, 80.0], score=0.99, label=0),
            ],
        }

        result = evaluator.evaluate(predictions)

        assert isinstance(result, DetectionMetrics)
        assert result.map_50 == pytest.approx(1.0, abs=1e-3)


class TestMapEvaluatorMultipleImages:
    """複数画像での mAP テスト."""

    def test_multiple_images_partial_match(self, tmp_path: Path) -> None:
        """複数画像で部分的に一致する場合の mAP を確認."""
        ann_path = _create_coco_annotation(
            tmp_path,
            images=[
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
            ],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]},
                {"id": 2, "image_id": 2, "category_id": 2, "bbox": [100, 100, 80, 80]},
            ],
        )

        evaluator = MapEvaluator(ann_path)
        predictions = {
            # img1: 完全一致
            "img1.jpg": [
                Detection(box=[10.0, 20.0, 60.0, 80.0], score=0.99, label=0),
            ],
            # img2: 全く異なる位置
            "img2.jpg": [
                Detection(box=[0.0, 0.0, 10.0, 10.0], score=0.99, label=1),
            ],
        }

        result = evaluator.evaluate(predictions)

        # img1 は完全一致, img2 は不一致 -> mAP は 0 と 1 の間
        assert 0.0 < result.map_50 < 1.0


class TestMapEvaluatorMissingPredictions:
    """GT に存在するが predictions にない画像が FN としてカウントされるテスト."""

    def test_missing_predictions_lower_map(self, tmp_path: Path) -> None:
        """predictions にない GT 画像が FN としてカウントされ mAP が下がることを確認."""
        ann_path = _create_coco_annotation(
            tmp_path,
            images=[
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
            ],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 20, 50, 60]},
            ],
        )

        evaluator = MapEvaluator(ann_path)

        # img1 のみ完全一致, img2 は predictions に含めない
        predictions = {
            "img1.jpg": [
                Detection(box=[10.0, 20.0, 60.0, 80.0], score=0.99, label=0),
            ],
        }

        result = evaluator.evaluate(predictions)

        # img2 の GT が FN としてカウントされるため mAP < 1.0
        assert result.map_50 < 1.0

    def test_all_predictions_missing_yield_map_zero(self, tmp_path: Path) -> None:
        """predictions が完全に空の場合 mAP が 0.0 になることを確認."""
        ann_path = _create_coco_annotation(
            tmp_path,
            images=[
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            ],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]},
            ],
        )

        evaluator = MapEvaluator(ann_path)
        predictions: dict[str, list[Detection]] = {}

        result = evaluator.evaluate(predictions)

        assert result.map_50 == pytest.approx(0.0, abs=1e-3)
        assert result.map_50_95 == pytest.approx(0.0, abs=1e-3)


class TestMapEvaluatorSubdirFilename:
    """サブディレクトリ付き file_name でのベースネームマッチテスト."""

    def test_basename_match_with_backslash(self, tmp_path: Path) -> None:
        """バックスラッシュ付きパスでベースネームマッチすることを確認."""
        ann_path = _create_coco_annotation(
            tmp_path,
            images=[
                {
                    "id": 1,
                    "file_name": "JPEGImages\\img1.jpg",
                    "width": 640,
                    "height": 480,
                },
            ],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]},
            ],
        )

        evaluator = MapEvaluator(ann_path)
        predictions = {
            "img1.jpg": [
                Detection(box=[10.0, 20.0, 60.0, 80.0], score=0.99, label=0),
            ],
        }

        result = evaluator.evaluate(predictions)

        assert result.map_50 == pytest.approx(1.0, abs=1e-3)

    def test_basename_match_with_slash(self, tmp_path: Path) -> None:
        """スラッシュ付きパスでベースネームマッチすることを確認."""
        ann_path = _create_coco_annotation(
            tmp_path,
            images=[
                {
                    "id": 1,
                    "file_name": "JPEGImages/img1.jpg",
                    "width": 640,
                    "height": 480,
                },
            ],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]},
            ],
        )

        evaluator = MapEvaluator(ann_path)
        predictions = {
            "img1.jpg": [
                Detection(box=[10.0, 20.0, 60.0, 80.0], score=0.99, label=0),
            ],
        }

        result = evaluator.evaluate(predictions)

        assert result.map_50 == pytest.approx(1.0, abs=1e-3)


class TestMapEvaluatorCategoryOrder:
    """カテゴリ ID の出現順に依存しないマッピングのテスト."""

    def test_reversed_category_order_gives_same_result(self, tmp_path: Path) -> None:
        """categories の出現順が逆でも同じ mAP が得られることを確認."""
        # カテゴリID昇順 (id=1 → idx=0, id=2 → idx=1)
        data_asc = {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]},
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"},
            ],
        }
        path_asc = tmp_path / "asc.json"
        path_asc.write_text(json.dumps(data_asc), encoding="utf-8")

        # カテゴリID降順 (id=2 が先)
        data_desc = {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]},
            ],
            "categories": [
                {"id": 2, "name": "dog"},
                {"id": 1, "name": "cat"},
            ],
        }
        path_desc = tmp_path / "desc.json"
        path_desc.write_text(json.dumps(data_desc), encoding="utf-8")

        predictions = {
            "img1.jpg": [
                Detection(box=[10.0, 20.0, 60.0, 80.0], score=0.99, label=0),
            ],
        }

        result_asc = MapEvaluator(path_asc).evaluate(predictions)
        result_desc = MapEvaluator(path_desc).evaluate(predictions)

        assert result_asc.map_50 == pytest.approx(result_desc.map_50, abs=1e-3)
        assert result_asc.map_50_95 == pytest.approx(result_desc.map_50_95, abs=1e-3)


class TestMapEvaluatorBasenameCollision:
    """basename 重複時の警告テスト."""

    def test_duplicate_basename_skips_registration(self, tmp_path: Path) -> None:
        """同名 basename が複数ある場合にフルパスマッチのみ使用されることを確認."""
        data = {
            "images": [
                {"id": 1, "file_name": "train/img.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "val/img.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 20, 50, 60]},
            ],
            "categories": [{"id": 1, "name": "cat"}],
        }
        path = tmp_path / "annotations.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        evaluator = MapEvaluator(path)

        # basename "img.jpg" は最初の image_id=1 のみ登録され, 2 番目は登録されない
        # (衝突時はスキップされる)
        assert evaluator._image_id_by_filename["img.jpg"] == 1
        # フルパスは両方登録されている
        assert evaluator._image_id_by_filename["train/img.jpg"] == 1
        assert evaluator._image_id_by_filename["val/img.jpg"] == 2


class TestExtractBasename:
    """_extract_basename のテスト."""

    def test_backslash_path(self) -> None:
        """バックスラッシュ付きパスからベースネームを抽出."""
        assert MapEvaluator._extract_basename("JPEGImages\\img.jpg") == "img.jpg"

    def test_slash_path(self) -> None:
        """スラッシュ付きパスからベースネームを抽出."""
        assert MapEvaluator._extract_basename("JPEGImages/img.jpg") == "img.jpg"

    def test_plain_filename(self) -> None:
        """パス区切りなしのファイル名はそのまま返す."""
        assert MapEvaluator._extract_basename("img.jpg") == "img.jpg"

    def test_nested_path(self) -> None:
        """ネストされたパスからベースネームを抽出."""
        assert MapEvaluator._extract_basename("a/b/c/img.jpg") == "img.jpg"


class TestXywhToXyxy:
    """_xywh_to_xyxy の変換テスト."""

    def test_conversion(self) -> None:
        """[x, y, w, h] -> [x1, y1, x2, y2] の変換を確認."""
        result = MapEvaluator._xywh_to_xyxy([10.0, 20.0, 50.0, 60.0])
        assert result == [10.0, 20.0, 60.0, 80.0]

    def test_zero_size(self) -> None:
        """サイズ 0 のボックスの変換を確認."""
        result = MapEvaluator._xywh_to_xyxy([5.0, 5.0, 0.0, 0.0])
        assert result == [5.0, 5.0, 5.0, 5.0]
