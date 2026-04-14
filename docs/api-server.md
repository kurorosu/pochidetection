# 推論 API クライアント使用例

pochidetection 推論 API サーバーにリクエストを送信するクライアントコードの例です.

## サーバー起動

```bash
# 学習済みモデル (PyTorch)
uv run pochi serve -m work_dirs/20260124_001/best

# ONNX モデル
uv run pochi serve -m work_dirs/20260124_001/best/model_fp32.onnx

# TensorRT エンジン
uv run pochi serve -m work_dirs/20260124_001/best/model_fp32.engine

# ホスト / ポート変更
uv run pochi serve -m work_dirs/20260124_001/best --host 0.0.0.0 --port 9000
```

起動時に設定ファイルはモデルパスから自動解決されます. 明示指定する場合は `-c configs/rtdetr_coco.py` を付与してください. 起動直後にダミー画像で warmup 推論が走るため, 1 回目のリクエストから安定したレイテンシで応答します.

## raw 形式 (numpy 配列)

cv2 でキャプチャした numpy 配列をそのまま送信する方式. ローカル環境向け.

```python
import base64

import numpy as np
import requests

# cv2 でキャプチャした画像 (H, W, 3) uint8 BGR
image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

payload = {
    "image_data": base64.b64encode(image.tobytes()).decode(),
    "shape": list(image.shape),
    "dtype": "uint8",
    "format": "raw",
    "score_threshold": 0.5,
}

response = requests.post("http://localhost:8000/api/v1/detect", json=payload)
result = response.json()

print(f"検出数: {len(result['detections'])}")
for det in result["detections"]:
    x1, y1, x2, y2 = det["bbox"]
    print(
        f"  {det['class_name']} (id={det['class_id']}) "
        f"conf={det['confidence']:.3f} bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
    )
print(f"E2E 処理時間: {result['e2e_time_ms']:.1f}ms (backend={result['backend']})")
```

## jpeg 形式 (圧縮転送)

JPEG 圧縮して送信する方式. ネットワーク越しの通信向け (データ量が約 1/20 に削減).

```python
import base64

import cv2
import requests

image = cv2.imread("test_image.jpg")  # BGR uint8
_, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])

payload = {
    "image_data": base64.b64encode(buf.tobytes()).decode(),
    "format": "jpeg",
    "score_threshold": 0.3,
}

response = requests.post("http://localhost:8000/api/v1/detect", json=payload)
print(response.json())
```

## pochidetection のシリアライザを使用

pochidetection に含まれるシリアライザを使うと, エンコード処理を省略できます.

```python
import numpy as np
import requests

from pochidetection.api.serializers import RawArraySerializer

image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

serializer = RawArraySerializer()
payload = serializer.serialize(image)
payload["score_threshold"] = 0.5

response = requests.post("http://localhost:8000/api/v1/detect", json=payload)
print(response.json())
```

## 検出結果を元画像に描画

レスポンスの bbox は**元画像座標系** `[x1, y1, x2, y2]` (ピクセル) のため, モデル入力解像度への座標変換は不要です.

```python
import cv2
import requests

from pochidetection.api.serializers import JpegSerializer

image = cv2.imread("test_image.jpg")
payload = JpegSerializer().serialize(image)
payload["score_threshold"] = 0.5

result = requests.post("http://localhost:8000/api/v1/detect", json=payload).json()

for det in result["detections"]:
    x1, y1, x2, y2 = map(int, det["bbox"])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{det['class_name']} {det['confidence']:.2f}"
    cv2.putText(image, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("test_image_result.jpg", image)
```

## 補助エンドポイント

```python
import requests

BASE = "http://localhost:8000/api/v1"

# ヘルスチェック
print(requests.get(f"{BASE}/health").json())
# {"status": "healthy", "model_loaded": true, "architecture": "RTDetr"}

# モデル情報
print(requests.get(f"{BASE}/model-info").json())
# {"architecture": "RTDetr", "num_classes": 4, "class_names": [...],
#  "input_size": [640, 640], "model_path": "...", "backend": "pytorch"}

# バージョン情報
print(requests.get(f"{BASE}/version").json())
# {"pochidetection_version": "0.15.0", "api_version": "v1",
#  "backend_versions": {"torch": "2.9.0", ...}}

# 利用可能バックエンド一覧
print(requests.get(f"{BASE}/backends").json())
# {"available": ["pytorch", "onnx", "tensorrt"], "current": "pytorch"}
```

## レスポンス形式

`POST /api/v1/detect` のレスポンスは以下の形式です.

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "pochi",
      "confidence": 0.95,
      "bbox": [100.5, 50.2, 200.8, 150.3]
    },
    {
      "class_id": 1,
      "class_name": "pochi2",
      "confidence": 0.72,
      "bbox": [300.0, 120.0, 380.0, 250.0]
    }
  ],
  "e2e_time_ms": 12.3,
  "backend": "tensorrt"
}
```

- `bbox`: `[x1, y1, x2, y2]` (左上・右下, 元画像座標系のピクセル値)
- `confidence`: 0.0-1.0
- `e2e_time_ms`: デシリアライズ後の `engine.predict()` を `time.perf_counter()` で計測した wall clock
- `backend`: `"pytorch"` / `"onnx"` / `"tensorrt"` のいずれか

## リクエスト仕様

### DetectRequest フィールド

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `image_data` | `str` | ✓ | base64 エンコードされた画像データ |
| `format` | `"raw"` / `"jpeg"` | (default `"raw"`) | 画像データ形式 |
| `shape` | `list[int]` | `format="raw"` 時のみ必須 | numpy 配列の shape `[height, width, 3]` |
| `dtype` | `str` | (default `"uint8"`) | numpy 配列の dtype (`uint8` のみ) |
| `score_threshold` | `float` (0-1) | (default `0.5`) | 検出信頼度の下限しきい値 |

- 最大画像サイズ: 4096 × 4096 ピクセル
- raw / jpeg のいずれも入力画像は **BGR** (cv2 convention) 前提. 内部で RGB に変換してパイプラインへ渡します
- dtype は `uint8` のみ対応 (float32 はセキュリティおよびパイプライン側の前処理仕様上未対応)

## エラーレスポンス

| ステータス | 発生条件 |
|---|---|
| `400` | 画像デシリアライズ失敗 (不正な base64, shape 不整合, JPEG デコード失敗 等) |
| `422` | リクエストスキーマバリデーションエラー (raw 形式で `shape` 欠落, `dtype` が `uint8` 以外, `score_threshold` 範囲外 等) |
| `500` | 推論中の予期しないエラー |
| `503` | モデル未ロード (warmup 中やシャットダウン中) |

## バックエンド自動判定

`pochi serve -m <path>` に渡したモデルファイルの拡張子でバックエンドが自動選択されます.

| 入力 | バックエンド |
|---|---|
| ディレクトリ / `.pth` | PyTorch |
| `.onnx` | ONNX Runtime (CUDA 可用時は `CUDAExecutionProvider`) |
| `.engine` | TensorRT |

`GET /api/v1/backends` の `current` で現在のバックエンドを確認できます.
