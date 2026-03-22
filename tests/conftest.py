"""テスト共通フィクスチャ."""

import json
from pathlib import Path

import pytest

from pochidetection.core.detection import Detection
from pochidetection.models import RTDetrModel, SSD300Model, SSDLiteModel
from pochidetection.utils import TrainingHistory


@pytest.fixture(scope="session")
def rtdetr_model() -> RTDetrModel:
    """テスト用の軽量RTDetrModelを作成するsessionスコープfixture.

    全テストで1つのインスタンスを共有し, モデル初期化コストを削減する.
    eval モードで返すため, train モードが必要なテストでは
    was_training パターンで一時切り替え・復元すること.
    """
    model = RTDetrModel(
        model_name="PekingU/rtdetr_r18vd", num_classes=2, pretrained=False
    )
    model.model.config.num_queries = 50
    model.eval()
    return model


@pytest.fixture(scope="session")
def ssd300_model() -> SSD300Model:
    """テスト用の軽量SSD300Modelを作成するsessionスコープfixture.

    全テストで1つのインスタンスを共有し, モデル初期化コストを削減する.
    eval モードで返すため, train モードが必要なテストでは
    was_training パターンで一時切り替え・復元すること.
    """
    model = SSD300Model(num_classes=2, pretrained=False)
    model.eval()
    return model


@pytest.fixture(scope="session")
def ssdlite_model() -> SSDLiteModel:
    """テスト用の軽量SSDLiteModelを作成するsessionスコープfixture.

    全テストで1つのインスタンスを共有し, モデル初期化コストを削減する.
    eval モードで返すため, train モードが必要なテストでは
    was_training パターンで一時切り替え・復元すること.
    """
    model = SSDLiteModel(num_classes=2, pretrained=False)
    model.eval()
    return model


@pytest.fixture(scope="class")
def single_epoch_history() -> TrainingHistory:
    """1エポック分の TrainingHistory を作成するfixture."""
    history = TrainingHistory()
    history.add(1, 0.5, 0.4, 0.3, 0.5, 0.2, 0.001)
    return history


@pytest.fixture()
def coco_annotation(tmp_path: Path) -> Path:
    """テスト用 COCO アノテーション (3画像, 2クラス, 4 GT).

    GT (xywh → xyxy):
    - img001: cat [10,20,90,180] → [10,20,100,200], dog [50,60,100,190] → [50,60,150,250]
    - img002: cat [5,10,75,110] → [5,10,80,120]
    - img003: cat [20,30,60,80] → [20,30,80,110]
    """
    ann = {
        "images": [
            {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img002.jpg", "width": 640, "height": 480},
            {"id": 3, "file_name": "img003.jpg", "width": 640, "height": 480},
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 90.0, 180.0],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [50.0, 60.0, 100.0, 190.0],
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [5.0, 10.0, 75.0, 110.0],
            },
            {
                "id": 4,
                "image_id": 3,
                "category_id": 1,
                "bbox": [20.0, 30.0, 60.0, 80.0],
            },
        ],
    }
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(ann), encoding="utf-8")
    return ann_path


@pytest.fixture()
def sample_predictions() -> dict[str, list[Detection]]:
    """テスト用推論結果 (coco_annotation の GT と完全一致するボックス).

    - img001: cat [10,20,100,200] (IoU=1.0), dog [50,60,150,250] (IoU=1.0)
    - img002: cat [5,10,80,120] (IoU=1.0)
    - img003: 検出なし
    """
    return {
        "img001.jpg": [
            Detection(box=[10.0, 20.0, 100.0, 200.0], score=0.95, label=0),
            Detection(box=[50.0, 60.0, 150.0, 250.0], score=0.80, label=1),
        ],
        "img002.jpg": [
            Detection(box=[5.0, 10.0, 80.0, 120.0], score=0.70, label=0),
        ],
        "img003.jpg": [],
    }
