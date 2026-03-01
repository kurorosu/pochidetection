# ONNX FP16 エクスポート調査報告

関連 Issue: #54

## 概要

RT-DETR モデルの ONNX FP16 エクスポートを試みたが, 現状利用可能な FP16 変換ツールでは正常に動作しないことが判明した.

## 検証環境

- Python 3.13
- onnx 1.20.1
- onnxruntime-gpu (CUDA)
- onnxconverter-common 1.16.0 / 1.13.0
- モデル: RT-DETR (HuggingFace Transformers)

## 試したアプローチ

### 1. onnxconverter-common の convert_float_to_float16()

**結果: 失敗**

```python
from onnxconverter_common.float16 import convert_float_to_float16

fp32_model = onnx.load("model.onnx")
fp16_model = convert_float_to_float16(fp32_model)
onnx.save(fp16_model, "model_fp16.onnx")
```

エラー:
```
onnxruntime.capi.onnxruntime_pybind11_state.Fail:
[ONNXRuntimeError] : 1 : FAIL : Load model from model_fp16.onnx failed:
Type Error: Type (tensor(float16)) of output arg
(/model/encoder/Cast_5_output_0) of node (/model/encoder/Cast_5)
does not match expected type (tensor(float)).
```

- v1.15.0 以降のリグレッション ([onnxruntime#25522](https://github.com/microsoft/onnxruntime/issues/25522))
- Cast ノードの出力型が不正に float16 に変換され, 後続ノードとの型不整合が発生
- `keep_io_types=True` や `op_block_list=["Cast"]` を試したが解消せず

### 2. onnxconverter-common==1.13.0 へのダウングレード

**結果: 失敗**

```
ModuleNotFoundError: No module named 'onnx.mapping'
```

- `onnx.mapping` モジュールは onnx 1.19 以降で削除済み
- onnxconverter-common 1.13.0 は onnx 1.20.1 と互換性なし

### 3. onnxruntime.transformers.float16 の convert_float_to_float16()

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

- onnxruntime.transformers 版も内部的に同じロジックを使用しており, 同様の問題が発生
- CUDA プロバイダーではモデルのロード・推論自体は可能だが, 60個の Memcpy ノードが自動挿入され, FP32 より大幅に遅くなる

## 結論

現時点では RT-DETR モデルに対して実用的な ONNX FP16 変換手段が存在しない.

### 根本原因

FP16 変換ツール (`onnxconverter-common` / `onnxruntime.transformers`) が RT-DETR の encoder 内部の Cast ノードを正しく扱えない. これはツール側のリグレッションであり, モデル側の問題ではない.

### 今後の対応案

1. **onnxconverter-common のバグ修正を待つ** - [onnxruntime#25522](https://github.com/microsoft/onnxruntime/issues/25522) の解決を待機
2. **PyTorch 側での FP16 エクスポート** - `torch.onnx.export` 時に FP16 モデル・入力を渡す方式を検討
3. **ONNX Runtime の Mixed Precision 機能** - `auto_mixed_precision` による部分的な FP16 化を検討
4. **TensorRT 経由の FP16 変換** - ONNX → TensorRT で FP16 推論を行う別アプローチ
