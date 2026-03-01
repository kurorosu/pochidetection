# RT-DETR 低精度推論 調査報告

関連 Issue: #54

## 概要

RT-DETR (HuggingFace Transformers) モデルの低精度 (FP16 / INT8) 推論の実現可能性を調査した.
ONNX レベルの FP16 変換と TensorRT による変換の両方を検証し, 現状の課題と今後の方針を整理する.

## 検証環境

- Python 3.13
- onnx 1.20.1
- onnxruntime-gpu (CUDA)
- onnxconverter-common 1.16.0 / 1.13.0
- モデル: RT-DETR (HuggingFace `transformers.RTDetrModel`)

---

## 1. ONNX レベルの FP16 変換

FP32 の ONNX グラフを直接 FP16 に書き換えるアプローチ.

### 1.1 onnxconverter-common (v1.16.0)

**結果: 失敗**

```python
from onnxconverter_common.float16 import convert_float_to_float16

fp32_model = onnx.load("model.onnx")
fp16_model = convert_float_to_float16(fp32_model)
onnx.save(fp16_model, "model_fp16.onnx")
```

エラー:
```
[ONNXRuntimeError] : 1 : FAIL : Load model from model_fp16.onnx failed:
Type Error: Type (tensor(float16)) of output arg
(/model/encoder/Cast_5_output_0) of node (/model/encoder/Cast_5)
does not match expected type (tensor(float)).
```

- v1.15.0 以降のリグレッション ([onnxruntime#25522](https://github.com/microsoft/onnxruntime/issues/25522))
- Cast ノードの出力型が不正に float16 に変換され, 後続ノードとの型不整合が発生
- `keep_io_types=True` や `op_block_list=["Cast"]` を試したが解消せず

### 1.2 onnxconverter-common (v1.13.0) へのダウングレード

**結果: 失敗**

```
ModuleNotFoundError: No module named 'onnx.mapping'
```

- `onnx.mapping` モジュールは onnx 1.19 以降で削除済み
- onnxconverter-common 1.13.0 は onnx 1.20.1 と互換性なし
- 互換性のある安全なバージョンが存在しない状態

### 1.3 onnxruntime.transformers.float16

**結果: 失敗**

```python
from onnxruntime.transformers.float16 import convert_float_to_float16
```

エラー:
```
Nodes in a graph must be topologically sorted, however input
'/model/encoder/Resize_input_cast_2' of node:
name: /model/encoder/Resize OpType: Resize
is not output of any previous nodes.
```

- onnxruntime.transformers 版も内部的に同じロジック (onnxconverter-common のフォーク) を使用
- CUDA プロバイダーではモデルのロード・推論自体は可能だが, 60 個の Memcpy ノードが自動挿入され FP32 より大幅に遅くなる

### ONNX FP16 変換の結論

現時点では RT-DETR モデルに対して実用的な ONNX レベルの FP16 変換手段が存在しない. ツール側のリグレッション (`onnxconverter-common` v1.15.0+) が根本原因であり, モデル側の問題ではない.

---

## 2. TensorRT による FP16 / INT8 変換

ONNX グラフを変更せず, TensorRT のエンジンビルド時に精度を指定するアプローチ.

### 2.1 TensorRT FP32

**結果: 問題なし**

```bash
trtexec --onnx=model_fp32.onnx --saveEngine=model_fp32.engine
```

- FP32 ONNX をそのまま TensorRT エンジンに変換可能
- 精度劣化なし

### 2.2 TensorRT FP16

**結果: 条件付きで可能 (Mixed Precision が必須)**

```bash
trtexec --onnx=model_fp32.onnx --fp16 --saveEngine=model_fp16.engine
```

ONNX グラフは一切変更されず, Cast ノード問題は完全に回避できる. ただし RT-DETR 固有の精度問題がある:

- **LayerNorm + Softmax の FP16 オーバーフロー**: Transformer の Attention 層で, Softmax 入力値が FP16 の最大値 (65504) を超えオーバーフロー → NaN が伝播
- **報告事例**: RF-DETR (同じ DETR 系) で mAP 0.69 → 0.29 に低下 ([rf-detr#176](https://github.com/roboflow/rf-detr/issues/176))
- **対策**: LayerNorm, Softmax, ReduceMean, Pow 等を FP32 に固定する Mixed Precision 設定が必須

```bash
trtexec --onnx=model.onnx --fp16 \
  --precisionConstraints=obey \
  --layerPrecisions=<layernorm_layer_names>:fp32
```

**追加のトレードオフ:**

| 項目 | 内容 |
|------|------|
| ポータビリティ | `.engine` ファイルは GPU / ドライバ / TRT バージョンに固定 |
| 依存関係 | TensorRT SDK の別途インストールが必要 |
| 初回レイテンシ | エンジンビルドに数秒〜数分かかる |
| Windows サポート | Linux より成熟度が低い |

### 2.3 TensorRT INT8

**結果: 現実的でない**

- Transformer の Attention 層の量子化が困難
- Post-Training Quantization (PTQ) では精度が大幅劣化, セグメンテーションフォールトの報告もあり
- 本格的な INT8 には Quantization-Aware Training (QAT) で別の重みを用意する必要がある
- 開発コストが非常に高く, 当面のスコープ外

### 2.4 ONNX Runtime TensorRT Execution Provider

TensorRT SDK がインストール済みであれば, ONNX Runtime 経由で TensorRT FP16 推論が可能:

```python
providers = [
    ('TensorrtExecutionProvider', {
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': './trt_cache',
    }),
    'CUDAExecutionProvider',  # フォールバック
]
sess = ort.InferenceSession("model_fp32.onnx", providers=providers)
```

入力 ONNX は FP32 のまま. エンジンビルドは初回実行時に自動で行われキャッシュされる. ただし LayerNorm の精度問題は同様に存在する.

---

## 総括

| アプローチ | FP32 | FP16 | INT8 |
|-----------|------|------|------|
| ONNX 変換 (`onnxconverter-common`) | - | 不可 (ツールのバグ) | - |
| TensorRT (naive `--fp16`) | 問題なし | 精度劣化 (LayerNorm オーバーフロー) | クラッシュ / 精度崩壊 |
| TensorRT (Mixed Precision) | 問題なし | **実装可能** (調整コスト高) | QAT 必須 (スコープ外) |
| ORT TensorRT EP | 問題なし | 実装可能 (TRT SDK 依存) | - |

### 推奨方針

1. **FP32**: 現状維持. ONNX Runtime (CUDA EP) で十分な性能が得られる
2. **FP16**: TensorRT + Mixed Precision で段階的に対応. まず FP32 TensorRT で動作確認し, LayerNorm 等を FP32 に固定した Mixed Precision へ進める
3. **INT8**: 当面見送り. QAT が必要で開発コストに見合わない

### 参考リンク

- [onnxruntime#25522 - Type mismatch error when loading Float16 model](https://github.com/microsoft/onnxruntime/issues/25522)
- [rf-detr#176 - TensorRT FP16 accuracy drop](https://github.com/roboflow/rf-detr/issues/176)
- [NVIDIA/TensorRT#1262 - FP16 large discrepancies for transformer detection](https://github.com/NVIDIA/TensorRT/issues/1262)
- [Float16 and mixed precision models - ONNX Runtime docs](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)
- [TensorRT Execution Provider - ONNX Runtime docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
