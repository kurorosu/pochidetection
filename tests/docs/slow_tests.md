# slow マーカー付きテスト一覧

`@pytest.mark.slow` を付与したテストは通常の `uv run pytest` では実行されない.

```bash
# 通常 (slow 除外, デフォルト)
uv run pytest

# slow のみ
uv run pytest -m slow

# 全テスト (slow 含む)
uv run pytest -m ""
```

## ファイル単位で slow (`pytestmark = pytest.mark.slow`)

| ファイル | テスト数 | 理由 |
|---------|---------|------|
| `test_cli/test_cli.py` | 5 | subprocess でプロセス起動するため 3-7s/テスト |
| `test_tensorrt/test_exporter.py` | 6 | GPU で TensorRT エンジンをビルドするため 2-3s/テスト |
| `test_tensorrt/test_ssdlite_trt_export.py` | 2 | 同上 |
| `test_onnx/test_rtdetr_exporter.py` | 7 | ONNX エクスポート + セッション作成で 0.5-0.8s/テスト |
| `test_onnx/test_rtdetr_onnx_backend.py` | 14 | 同上 |
| `test_onnx/test_ssdlite_exporter.py` | 11 | 同上 |
| `test_onnx/test_ssdlite_onnx_backend.py` | 15 | 同上 |

## 個別テストで slow (`@pytest.mark.slow`)

| ファイル | テスト名 | 理由 |
|---------|---------|------|
| `test_models/test_rtdetr_model.py` | `test_custom_num_classes` | `num_classes=10` で RTDetrModel を新規初期化 (~0.7s) |
| `test_models/test_rtdetr_model.py` | `test_load_updates_num_classes` | `num_classes=5` と `num_classes=2` の 2 つを初期化 + save/load (~1.5s) |
| `test_models/test_ssd300_model.py` | `test_num_classes_property` | `num_classes=4` で SSD300Model を新規初期化 (~0.9s) |
| `test_models/test_ssd300_model.py` | `test_nms_iou_threshold_custom` | `nms_iou_threshold=0.3` で SSD300Model を新規初期化 (~0.9s) |
| `test_models/test_ssdlite_model.py` | `test_num_classes_property` | `num_classes=4` で SSDLiteModel を新規初期化 |
| `test_models/test_ssdlite_model.py` | `test_nms_iou_threshold_custom` | `nms_iou_threshold=0.3` で SSDLiteModel を新規初期化 |

## session フィクスチャ (`conftest.py`)

モデル初期化コスト削減のため, 以下の session スコープフィクスチャを提供:

| フィクスチャ名 | 型 | パラメータ |
|-------------|---|---------|
| `rtdetr_model` | `RTDetrModel` | `num_classes=2, pretrained=False` |
| `ssd300_model` | `SSD300Model` | `num_classes=2, pretrained=False` |
| `ssdlite_model` | `SSDLiteModel` | `num_classes=2, pretrained=False` |

session フィクスチャと異なるパラメータ (`num_classes=4` 等) が必要なテストは, テスト内で個別に初期化するため slow マーカーを付与している.
