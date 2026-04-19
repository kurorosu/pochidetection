# pochidetection

[![Version](https://img.shields.io/badge/version-0.16.1-blue.svg)](https://github.com/kurorosu/pochidetection)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/docs/transformers)
[![torchvision](https://img.shields.io/badge/torchvision-SSDLite-ee4c2c.svg)](https://pytorch.org/vision/)

**pochi_series 物体検出フレームワーク**

## 特徴

- **マルチアーキテクチャ対応**: RT-DETR (Transformer), SSDLite (CNN), SSD300 (CNN) を設定ファイルで切り替え可能
- **COCO フォーマット対応**: 標準的なアノテーション形式でデータセットを管理
- **画像・動画・リアルタイム推論**: 画像, 動画ファイル, Webcam/RTSP ストリームに対応
- **COCO プリトレイン**: モデル未指定時は RT-DETR COCO プリトレインモデルで即座に推論可能
- **学習の可視化**: Loss 曲線, mAP 曲線, PR 曲線を HTML で自動出力. TensorBoard 連携にも対応 (`enable_tensorboard = True`)
- **Early Stopping**: mAP または val_loss を監視し, 改善がなければ学習を自動停止
- **CLI ツール**: コマンドひとつで学習・推論・エクスポートを実行
- **WebAPI サーバー**: FastAPI ベースの推論 API (`pochi serve`) で base64 画像から検出結果を REST 取得

## クイックスタート

### 1. インストール

```bash
uv sync
```

> GPU (TensorRT 等) を使用するためのネイティブ環境構築については, pochitrainの[GPU Environment Setup](https://github.com/kurorosu/pochitrain/blob/dev/pochitrain/docs/gpu_environment_setup.md) ドキュメントを参照してください.

> GPU を使用する場合は PyTorch の CUDA 対応版が自動でインストールされます.

### 2. データの準備

COCO フォーマットのディレクトリ構成でデータを準備してください:

```
data/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── annotations.json
└── val/
    ├── images/
    │   ├── img_100.jpg
    │   └── ...
    └── annotations.json
```

### 3. 設定ファイルの編集

`architecture` フィールドは大文字小文字を問わず指定できます (例: `"rtdetr"`, `"RTDETR"` → `"RTDetr"` に正規化).

#### RT-DETR (Transformer ベース)

`configs/rtdetr_coco.py` を編集してください:

```python
architecture = "RTDetr"
model_name = "PekingU/rtdetr_r50vd"
num_classes = 4
class_names = ["pochi", "pochi2", "pochi3", "pochi4"]
image_size = {"height": 640, "width": 640}
batch_size = 8
epochs = 5
learning_rate = 1e-4
device = "cuda"
```

#### SSDLite (CNN ベース, 軽量)

`configs/ssdlite_coco.py` を編集してください:

```python
architecture = "SSDLite"
model_name = "ssdlite320_mobilenet_v3_large"
num_classes = 4
class_names = ["pochi", "pochi2", "pochi3", "pochi4"]
image_size = {"height": 320, "width": 320}
batch_size = 16
epochs = 100
learning_rate = 1e-3
device = "cuda"
```

### 4. 学習の実行

```bash
# RT-DETR で学習 (デフォルト)
uv run pochi train -c configs/rtdetr_coco.py

# SSDLite で学習
uv run pochi train -c configs/ssdlite_coco.py
```

Data Augmentation は config.py の `augmentation` セクションで設定可能 (詳細は [docs/augmentation.md](docs/augmentation.md) を参照).

### 5. 結果の確認

学習結果は `work_dirs/yyyymmdd_xxx/` に保存されます:

| ファイル | 説明 |
|----------|------|
| `best/` | 最高 mAP のモデル |
| `last/` | 最終エポックのモデル |
| `training_history.csv` | エポックごとのメトリクス |
| `loss.html` | 損失曲線 (train vs val) |
| `metrics.html` | mAP 曲線 (mAP, mAP@50, mAP@75) |
| `pr_curve.html` | PR 曲線 (クラス別 + 平均) |
| `tensorboard/` | TensorBoard ログ (`enable_tensorboard = True` 時のみ) |

設定ファイルは元のファイル名を保持してワークスペースにコピーされます.

config.py で `enable_tensorboard = True` を設定すると, 各エポックの train/val loss, mAP (mAP / mAP@50 / mAP@75), 学習率を TensorBoard に記録します. 学習中・終了後ともに以下で確認できます:

```bash
tensorboard --logdir work_dirs/yyyymmdd_xxx/tensorboard
```

### 6. 推論の実行

#### 画像推論

`-m` でモデルディレクトリを指定すると, ワークスペース内の `config.py` が自動解決されます.
`-d` を省略した場合は config の `infer_image_dir` が使用されます.

```bash
# モデル指定 (config はワークスペースから自動解決)
uv run pochi infer -m work_dirs/20260124_001/best

# 画像ディレクトリも明示指定
uv run pochi infer -d images/ -m work_dirs/20260124_001/best

# ONNX モデルで推論
uv run pochi infer -m work_dirs/20260124_001/best/model_fp32.onnx

# config を明示指定 (自動解決より優先)
uv run pochi infer -d images/ -c configs/ssdlite_coco.py
```

推論結果は `work_dirs/yyyymmdd_xxx/inference_xxx/` に保存されます.

- バウンディングボックス付きの結果画像 (`{filename}_result.{ext}`)
- 検出ボックスのクロップ画像 (`crop/` フォルダ, デフォルト有効, `--no-crop` で無効化)
- 推論速度の統計 (平均 ms/image, 合計時間)

#### 動画推論

```bash
# 動画ファイルを指定して推論
uv run pochi infer -d video.mp4 -m work_dirs/20260124_001/best

# 3フレーム間隔で推論 (処理時間を短縮)
uv run pochi infer -d video.mp4 -m work_dirs/20260124_001/best --interval 3

# COCO プリトレインモデルで推論 (モデル未指定時)
uv run pochi infer -d video.mp4
```

- 対応形式: `.mp4`, `.avi`, `.mov`
- 出力: 入力と同じディレクトリに `{filename}_result.mp4` として保存
- `--interval N`: N フレーム間隔で推論 (スキップフレームはそのまま出力)
- モデル未指定時は RT-DETR COCO プリトレインモデルで推論

#### リアルタイム推論 (Webcam / RTSP)

```bash
# Webcam (デバイス ID 指定)
uv run pochi infer -d 0

# RTSP ストリーム
uv run pochi infer -d rtsp://192.168.1.100:554/stream

# 表示 + 録画 (推論フォルダに自動保存)
uv run pochi infer -d 0 --record

# モデル指定 + 3フレーム間隔で推論
uv run pochi infer -d 0 -m work_dirs/20260124_001/best --interval 3
```

- フェーズ別 FPS 内訳オーバーレイが自動表示 (capture/pre/infer/post/draw/display)
- `q` キーで終了, `Ctrl+C` でも安全に停止
- `s` キーでカメラ設定ダイアログを表示 (Windows, Webcam のみ)
- `--record`: 推論フォルダに `recording.mp4` として録画
- 推論完了時に `stream_metadata.json` (カメラ設定, FPS サマリー) を自動保存
- config.py で `camera_fps`, `camera_resolution` を指定可能
- モデル未指定時は RT-DETR COCO プリトレインモデルで推論

### 7. ONNX エクスポート

```bash
# RT-DETR: ONNX エクスポート (FP32)
uv run pochi export -m work_dirs/20260124_001/best

# SSDLite: ONNX エクスポート (FP32)
uv run pochi export -m work_dirs/20260124_001/best -c configs/ssdlite_coco.py

# SSDLite: ONNX エクスポート (FP16)
uv run pochi export -m work_dirs/20260124_001/best -c configs/ssdlite_coco.py --fp16

# 出力パスを指定
uv run pochi export -m work_dirs/20260124_001/best -o model.onnx

# 検証をスキップ
uv run pochi export -m work_dirs/20260124_001/best --skip-verify
```

`--fp16` は SSDLite のみ対応. エクスポート後, PyTorch と ONNX の出力が一致するか自動検証されます.

### 8. ONNX モデルで推論

```bash
# ONNX モデルファイルを直接指定して推論
uv run pochi infer -d images/ -m work_dirs/20260124_001/best/model_fp32.onnx -c configs/ssdlite_coco.py

# FP16 ONNX モデルで推論
uv run pochi infer -d images/ -m work_dirs/20260124_001/best/model_fp16.onnx -c configs/ssdlite_coco.py
```

ONNX ファイル (`.onnx`) を指定すると, 自動的に ONNX Runtime バックエンドが選択されます.
CUDA 環境では `CUDAExecutionProvider` が使用されます.

### 9. TensorRT エクスポート

```bash
# ONNX → TensorRT エクスポート (FP32)
uv run pochi export -m model.onnx

# TensorRT エクスポート (FP16)
uv run pochi export -m model.onnx --fp16

# TensorRT エクスポート (INT8, Post-Training Quantization)
uv run pochi export -m model.onnx --int8

# INT8 キャリブレーション画像数を制限
uv run pochi export -m model.onnx --int8 --calib-max-images 100

# ビルド時メモリ制限を指定 (デフォルト: 4GB)
uv run pochi export -m model.onnx --build-memory 2147483648
```

`export` コマンドは `-m` に渡すパスで動作を自動判定します:
- フォルダ → ONNX エクスポート
- `.onnx` ファイル → TensorRT エクスポート

INT8 キャリブレーション画像は config の `infer_image_dir` から取得されます.

### 10. WebAPI サーバー (`pochi serve`)

FastAPI + uvicorn ベースの推論 API サーバーを起動し, base64 エンコードされた画像を POST して検出結果 (bbox, class, confidence) を取得できます.

```bash
# PyTorch モデル (default: --pipeline gpu)
uv run pochi serve -m work_dirs/20260124_001/best

# ONNX / TensorRT モデル (拡張子でバックエンド自動判定)
uv run pochi serve -m work_dirs/20260124_001/best/model_fp32.onnx
uv run pochi serve -m work_dirs/20260124_001/best/model_fp32.engine

# preprocess 経路を CPU に強制 (ベンチマーク比較用)
uv run pochi serve -m work_dirs/20260124_001/best --pipeline cpu

# 動作確認
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/model-info
```

主なエンドポイント:

| メソッド | パス | 用途 |
|---|---|---|
| `POST` | `/api/v1/detect` | 画像から検出結果を取得 (bbox, class, confidence) |
| `GET` | `/api/v1/health` | サーバー / モデル状態 |
| `GET` | `/api/v1/model-info` | アーキテクチャ・クラス名・入力サイズ |
| `GET` | `/api/v1/version` | pochidetection / backend ライブラリのバージョン |
| `GET` | `/api/v1/backends` | 利用可能 / 現在のバックエンド |

`POST /api/v1/detect` のレスポンスには `e2e_time_ms` (router 全体の wall) と `phase_times_ms` (`pipeline_preprocess_ms` / `pipeline_inference_ms` / `pipeline_postprocess_ms` + CUDA 利用時 `pipeline_inference_gpu_ms`) が含まれ, INFO ログには 1 行サマリ + GPU クロック (`clk=YYYYMHz`) + 採用経路 (`pipeline=cpu/gpu`) が併記されます.

`--pipeline cpu/gpu` の挙動:

| backend | `--pipeline` 未指定 | `--pipeline cpu` | `--pipeline gpu` |
|---|---|---|---|
| PyTorch / TensorRT | `gpu` (default) | `cpu` | `gpu` |
| ONNX | `cpu` (自動) | `cpu` | **起動エラー** (ONNX Runtime は CPU numpy 入力のため) |

リクエスト例, レスポンス形式, エラーハンドリングの詳細は [docs/api-server.md](docs/api-server.md) を参照してください.

## サポート機能

### モデル

| モデル | 設定値 | 説明 |
|--------|--------|------|
| RT-DETR (R50) | `architecture = "RTDetr"` | HuggingFace Pretrained Transformer モデル (640x640) |
| SSDLite MobileNetV3 | `architecture = "SSDLite"` | torchvision 軽量 CNN モデル (320x320) |
| SSD300 VGG16 | `architecture = "SSD300"` | torchvision CNN モデル (300x300) |

### 評価指標

- mAP (IoU=0.50:0.95)
- mAP@50
- mAP@75
- クラス別 Precision-Recall 曲線

### 高度な機能

- **Early Stopping**: `early_stopping_patience = 10` で mAP/val_loss の改善を監視し自動停止
- **Learning Rate Scheduler**: `lr_scheduler = "CosineAnnealingLR"` 等の PyTorch 標準スケジューラ対応
- **FP16 混合精度**: CUDA 環境で `use_fp16 = True` により高速推論
- **cuDNN Benchmark**: `cudnn_benchmark = True` で GPU 推論を最適化
- **自動ワークスペース管理**: `work_dirs/yyyymmdd_xxx/` で学習結果を自動管理
- **インタラクティブ可視化**: Plotly による HTML グラフで学習過程を分析
- **TensorBoard 連携**: `enable_tensorboard = True` で各エポックの train/val loss, mAP (mAP / mAP@50 / mAP@75), 学習率を `work_dirs/<workspace>/tensorboard/` に記録. `tensorboard --logdir` で可視化可能
- **動画推論**: OpenCV による動画ファイルのフレーム単位推論. `--interval` でフレーム間隔指定可能
- **リアルタイム推論**: Webcam (`-d 0`) / RTSP (`-d rtsp://...`) ストリーム対応. フェーズ別 FPS 内訳表示, `--record` で推論フォルダに録画, `s` キーでカメラ設定, config.py で FPS・解像度設定可能
- **COCO プリトレイン推論**: モデル未指定時に RT-DETR COCO プリトレインモデルで即座に推論
- **ONNX エクスポート**: RT-DETR / SSDLite 両対応. SSDLite は FP16 エクスポートにも対応
- **ONNX 推論**: ONNX Runtime による推論バックエンド (CUDA / CPU 自動選択)
- **TensorRT エクスポート**: ONNX モデルから TensorRT エンジンへの変換 (FP32/FP16/INT8, Dynamic Batching 対応)
- **TensorRT 推論**: RT-DETR / SSDLite 両対応. `.engine` ファイル指定で自動選択
- **INT8 Post-Training Quantization**: `INT8Calibrator` によるキャリブレーション付き INT8 エンジンビルド
- **WebAPI サーバー**: `pochi serve` で FastAPI + uvicorn 起動. `POST /api/v1/detect` で base64 (raw / jpeg) 画像から検出結果を返却. `score_threshold` 指定可, PyTorch / ONNX / TensorRT 3 バックエンドを拡張子で自動選択
- **GPU preprocess 経路**: `--pipeline cpu/gpu` で起動時切替 (PyTorch / TensorRT は default `gpu`). uint8 H2D + GPU 上 `[0,1]` 化 + 入力バッファ再利用で preprocess を 7-12ms → 3-4ms に短縮. CLI / カメラ / WebAPI 全経路で効果. ONNX backend は CPU 経路のみ対応
- **CUDA Event 計測 + GPU クロック表示**: `/detect` の inference を CUDA Event で計測し `pipeline_inference_gpu_ms` を返却. INFO ログに pynvml 由来の GPU クロック (`clk=YYYYMHz`) を併記し adaptive clock policy による振動を可視化

## 注意点

- 対応画像形式: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`
- 対応動画形式: `.mp4`, `.avi`, `.mov`
- 推論では最初の画像がウォームアップとして計測から除外されます
- COCO アノテーションの座標は自動的に正規化 `[cx, cy, w, h]` 形式に変換されます
- バックグラウンドクラス (category_id=0) は自動的に除外されます
- `export` コマンドは RT-DETR と SSDLite の両方に対応. `-m` にフォルダを指定すると ONNX, `.onnx` を指定すると TensorRT エクスポートを自動実行

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています.
