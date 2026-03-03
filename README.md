# pochidetection

[![Version](https://img.shields.io/badge/version-0.4.2-blue.svg)](https://github.com/kurorosu/pochidetection)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/docs/transformers)

A tiny but clever object detection pipeline — as friendly as Pochi!

**pochi_series の設計思想に基づいた, Python 3.13+ 向け物体検出フレームワーク**

## 🎯 特徴

- **HuggingFace Transformers ベース**: RT-DETR などの最新モデルをすぐに利用可能
- **COCO フォーマット対応**: 標準的なアノテーション形式でデータセットを管理
- **学習の可視化**: Loss 曲線, mAP 曲線, PR 曲線を HTML で自動出力
- **CLI ツール**: コマンドひとつで学習・推論を実行

## 🚀 クイックスタート

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

`configs/rtdetr_coco.py` を編集してください:

```python
# モデル
model_name = "PekingU/rtdetr_r50vd"
num_classes = 4
class_names = ["pochi", "pochi2", "pochi3", "pochi4"]

# 画像サイズ
image_size = {"height": 640, "width": 640}

# データ
data_root = "data"
train_split = "train"
val_split = "val"
batch_size = 8

# 学習
epochs = 5
learning_rate = 1e-4

# デバイス
device = "cuda"
use_fp16 = False
```

### 4. 学習の実行

```bash
# デフォルト設定で学習
uv run pochi train

# 設定ファイルを指定して学習
uv run pochi train -c configs/rtdetr_coco.py
```

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

### 6. 推論の実行

```bash
# 画像ディレクトリを指定して推論
uv run pochi infer -d images/

# 信頼度閾値を指定
uv run pochi infer -d images/ -t 0.3

# 学習済みモデルを指定
uv run pochi infer -d images/ -m work_dirs/20260124_001/best
```

推論結果は `work_dirs/yyyymmdd_xxx/inference_xxx/` に保存されます.

- バウンディングボックス付きの結果画像 (`{filename}_result.{ext}`)
- 推論速度の統計 (平均 ms/image, 合計時間)

## 🛠️ サポート機能

### モデル

| モデル | 説明 |
|--------|------|
| RT-DETR (R50) | `PekingU/rtdetr_r50vd` - HuggingFace Pretrained |

### 評価指標

- mAP (IoU=0.50:0.95)
- mAP@50
- mAP@75
- クラス別 Precision-Recall 曲線

### 高度な機能

- **FP16 混合精度**: CUDA 環境で `use_fp16 = True` により高速学習・推論
- **cuDNN Benchmark**: `cudnn_benchmark = True` で GPU 推論を最適化
- **自動ワークスペース管理**: `work_dirs/yyyymmdd_xxx/` で学習結果を自動管理
- **インタラクティブ可視化**: Plotly による HTML グラフで学習過程を分析

## 📝 注意点

- 対応画像形式: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`
- 推論では最初の画像がウォームアップとして計測から除外されます
- COCO アノテーションの座標は自動的に正規化 `[cx, cy, w, h]` 形式に変換されます
- バックグラウンドクラス (category_id=0) は自動的に除外されます

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています.
