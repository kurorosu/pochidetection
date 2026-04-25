# SSDLite ONNX FP16 速度改善なし 調査報告

関連 Issue: [#225](https://github.com/kurorosu/pochidetection/issues/225)

## 概要

SSDLite ONNX FP16 モデルが FP32 比で速度改善を示さない問題を調査した.
結論として, MobileNetV3 の Depthwise Separable Convolution が FP16 / Tensor Core に不向きであるという **アーキテクチャ由来の根本的制約** が原因であり, コードの不具合ではない.

## 計測結果

| Configuration | Avg Inference Time (ms) |
|---|---|
| ONNX FP32 | 7.45 |
| ONNX FP16 | 7.27 |

差は 0.18ms (2.4%) で, 実質的に同等.

---

## 1. 根本原因: Depthwise Separable Convolution と Tensor Core の不適合

### Tensor Core の要件

NVIDIA Tensor Core は行列演算を高速化するが, 以下の要件がある:

- 行列次元が **8 の倍数** (FP16) であること
- 効率的な動作には **64-256 以上のチャンネル数** が必要
- **NHWC レイアウト** が推奨

### MobileNetV3 の Depthwise Conv が不適合な理由

| 要因 | 詳細 |
|------|------|
| チャンネル次元 | Depthwise conv はチャンネル次元が 1. Tensor Core の最低要件 (8) を満たせない |
| 小さいチャンネル数 | MobileNetV3 は 16ch 程度の層がある. Tensor Core 効率は 64-256ch で最大化 |
| 演算密度 | 軽量モデルは計算がボトルネックではなく, メモリ帯域がボトルネック. FP16 の計算量半減が効かない |

### 既知の事例

| Model | Hardware | FP32 | FP16 | Speedup |
|-------|----------|------|------|---------|
| MobileNetV3 | RTX 3090 | 169 fps | 168 fps | **0% (なし)** |
| ResNet50 (standard conv) | RTX 3090 | -- | -- | **~15%** |
| RFB-Ghost (depthwise) | TensorRT GPU | 4.9 fps | 5.3 fps | **8%** |
| RFB-VGG (standard conv) | TensorRT GPU | 4.3 fps | 6.7 fps | **56%** |

Sources:
- [RobustVideoMatting#94](https://github.com/PeterL1n/RobustVideoMatting/issues/94): MobileNetV3 fp16 no speedup
- [TensorRT#444](https://github.com/NVIDIA/TensorRT/issues/444): Depthwise conv FP16 speedup ratio
- [TensorFlow#27285](https://github.com/tensorflow/tensorflow/issues/27285): Depthwise separable conv FP16 slower

### Depthwise Conv + FP16 の悪化ケース

TensorFlow の報告では, depthwise separable convolution が FP16 で **1.6x 遅くなる** ケースが確認されている.
GPU カーネルが 32-bit データを前提とした memoryalignment を使っており, FP16 ではアラインメント不整合が発生するため.

---

## 2. ONNX Runtime 側の問題

### 2.1 InsertedPrecisionFreeCast ノード

ONNX Runtime は FP16 カーネルが存在しない演算子に対して, 自動的に `InsertedPrecisionFreeCast` ノードを挿入する.
このノードは FP16→FP32→FP16 のキャストを行い, オーバーヘッドとなる.

- [onnxruntime#25824](https://github.com/microsoft/onnxruntime/issues/25824): Cast ノードが **推論時間の ~54%** を占めるケース
- 小さいモデルほどキャストのオーバーヘッドが相対的に大きくなる

### 2.2 現行コードの SessionOptions 未設定

現在の `SSDLiteOnnxBackend` は `ort.InferenceSession()` に対して `SessionOptions` を渡していない:

```python
# pochidetection/inference/ssdlite/onnx_backend.py:91
self._session = ort.InferenceSession(str(model_path), providers=providers)
```

以下の設定が未適用:

| Option | Default | FP16 への影響 |
|--------|---------|--------------|
| `cudnn_conv_use_max_workspace` | `"1"` (v1.14+) | cuDNN が Tensor Core アルゴリズムを選択可能にする. デフォルトで有効 |
| `cudnn_conv_algo_search` | `EXHAUSTIVE` | 全アルゴリズムをベンチマーク. デフォルトで最適 |
| `prefer_nhwc` | `"0"` | Tensor Core は NHWC が効率的. 未設定 |
| `enable_cuda_graph` | `"0"` | カーネル起動オーバーヘッド削減. IOBinding 必須 |

ただし, **Depthwise Conv が Tensor Core を使えない** 以上, これらの設定変更による改善は限定的.

### 2.3 IOBinding 未使用

現在の `infer()` は毎回 CPU↔GPU 間のデータ転送が発生:

```python
# onnx_backend.py:186
tensor = inputs[name].cpu()  # GPU→CPU
numpy_inputs[name] = tensor.half().numpy()
# ...
raw_outputs = self._session.run(None, numpy_inputs)  # CPU→GPU→CPU
```

IOBinding を使えばデータを GPU 上に保持でき, 転送オーバーヘッドを削減可能.
ただし, 7ms 程度の推論で転送オーバーヘッドは 0.1-0.5ms 程度であり, FP16 速度改善の本質的解決にはならない.

---

## 3. ONNX モデルの精度検証方法

FP16 モデルが本当に FP16 で演算されているか確認する方法:

### プロファイリング

```python
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True
sess_options.optimized_model_filepath = "optimized_fp16.onnx"

session = ort.InferenceSession("model_fp16.onnx", sess_options, providers=[...])
```

- `optimized_model_filepath` で最適化後のグラフを保存し, Netron で `InsertedPrecisionFreeCast` ノードの有無を確認可能
- プロファイリング JSON を Perfetto UI (`https://ui.perfetto.dev/`) で可視化し, 演算子別の実行時間を分析可能

### ノード精度検査

```python
import onnx
from onnx import TensorProto

model = onnx.load("model.onnx")
dtypes = {init.data_type for init in model.graph.initializer}
print("Weight dtypes:", {TensorProto.DataType.Name(d) for d in dtypes})
# {'FLOAT'} → 重みが FP32 (真の FP16 ではない)
# {'FLOAT16'} → 重みが FP16
```

---

## 4. RT-DETR ONNX Backend との比較

| 観点 | RT-DETR | SSDLite |
|------|---------|---------|
| FP16 入力サポート | なし (FP32 に固定) | あり (自動検出) |
| SessionOptions | 未設定 | 未設定 |
| IOBinding | 未使用 | 未使用 |
| FP16 エクスポート | 未対応 | 対応済み |
| 後処理 | Pipeline 側で実施 | Backend 内で実施 |

両バックエンドとも SessionOptions と IOBinding が未設定という共通点がある.

---

## 5. 結論と推奨方針

### 結論

SSDLite (MobileNetV3) で FP16 の速度改善がないのは **アーキテクチャ由来の根本的制約**:

1. **Depthwise Separable Convolution** が Tensor Core に不適合
2. **小さいチャンネル数** が Tensor Core の効率要件を満たさない
3. **軽量モデル** はメモリ帯域がボトルネックで, FP16 の計算量半減が効かない
4. **FP16↔FP32 キャスト** のオーバーヘッドが小さいモデルでは相対的に大きい

### FP16 の価値

速度改善はないが, **モデルサイズが約 50% 削減** される点は引き続き有効.
ストレージやモデル配信のコスト削減に寄与する.

### 今後の最適化パス

| 優先度 | アプローチ | 期待効果 | 備考 |
|--------|-----------|----------|------|
| 1 | IOBinding 導入 | ~0.3ms 改善 | FP32/FP16 共通で効果あり |
| 2 | SessionOptions 最適化 | 軽微 | `prefer_nhwc`, graph optimization |
| 3 | INT8 量子化 | **2-3x 高速化** | MobileNet 系で最も効果的. 精度影響の検証が必要 |
| 4 | TensorRT バックエンド | 速度改善 | SSDLite 用 TensorRT エクスポーター未実装 |

### 参考リンク

- [NVIDIA: Tips for Optimizing GPU Performance Using Tensor Cores](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/)
- [ONNX Runtime Float16 and mixed precision models](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)
- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)
- [onnxruntime#13838: FP16 Conv slower than FP32](https://github.com/microsoft/onnxruntime/issues/13838)
- [onnxruntime#25824: FP16 performance bottleneck with Cast nodes](https://github.com/microsoft/onnxruntime/issues/25824)
