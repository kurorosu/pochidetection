# SSDLite ONNX 推論 mAP 低下 調査報告

関連 Issue: #224

## 概要

SSDLite ONNX 推論で PyTorch 比 mAP が約 5% 低下する問題の原因を調査した.
FP32 と FP16 の mAP がほぼ同一であることから, ONNX エクスポートの数値精度ではなく前処理の差異が原因と推測し, コードレベルの詳細分析を実施した.

---

## ベンチマーク結果

| | PyTorch | ONNX FP32 | ONNX FP16 |
|---|---|---|---|
| mAP@0.5 | **0.5606** | 0.5095 | 0.5095 |
| mAP@0.5:0.95 | **0.2380** | 0.2067 | 0.2066 |

FP32 と FP16 の mAP がほぼ同一 → ONNX エクスポートの数値精度は問題なし.

---

## 根本原因: 二重正規化バグ

### torchvision SSDLite の設計意図

torchvision の `ssdlite320_mobilenet_v3_large` は以下の設計:

- 入力: [0, 1] の浮動小数点画像
- `GeneralizedRCNNTransform(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])` が `SSD.forward()` 内部で自動適用
- backbone (MobileNetV3) は [-1, 1] の入力を期待

ソースコード (`torchvision/models/detection/ssdlite.py` 行308-317):

```python
defaults = {
    # Rescale the input in a way compatible to the backbone:
    # The following mean/std rescale the data from [0, 1] to [-1, 1]
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}
```

### 現状のバグ

pochidetection では, Dataset (`SsdCocoDataset`) と推論 Pipeline (`infer.py`) が ImageNet 正規化を適用している:

```python
# ssd_coco_dataset.py (行47-57) / infer.py (行164-173)
v2.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
```

この後に torchvision SSD 内部で `GeneralizedRCNNTransform` がさらに適用されるため, 二重正規化が発生:

```
[0, 255] → ToDtype(scale=True) → [0, 1]
       → ImageNet Normalize → 約 [-2, 2]  ← 不要な正規化
       → GeneralizedRCNNTransform → 約 [-5, 3]  ← 設計上は [-1, 1] であるべき
       → backbone
```

---

## 処理フロー比較

### PyTorch パス (学習時・推論時共通)

```
画像 [0,255]
  → ToDtype(scale=True)                 → [0, 1]
  → Normalize([0.485..], [0.229..])     → ImageNet正規化済み ← バグ
  → SSDLiteModel.forward()
    → torchvision SSD.forward()
      → GeneralizedRCNNTransform([0.5], [0.5]) → 二重正規化
  → backbone + head + postprocess
```

### ONNX パス (推論時)

```
画像 [0,255]
  → ToDtype(scale=True)                 → [0, 1]
  → Normalize([0.485..], [0.229..])     → ImageNet正規化済み ← バグ
  → SSDLiteOnnxBackend.infer()
    → ONNX Runtime (backbone + head のみ)
      → GeneralizedRCNNTransform なし ← 欠落
  → _postprocess
```

### PyTorch で mAP が出ている理由

学習時も推論時も同じ二重正規化が適用されるため, モデルは「二重正規化された入力分布」で学習しており, 推論時も同じ入力分布になるため整合が取れている.

### ONNX で mAP が低下する理由

`_SSDLiteExportWrapper` が backbone + head のみをラップし, `GeneralizedRCNNTransform` をスキップしている. その結果, ONNX backbone に入力される値は「ImageNet正規化のみ」であり, 学習時の「ImageNet正規化 + GeneralizedRCNNTransform」とは異なる入力分布になっている.

---

## _SSDLiteExportWrapper の構造

```python
# ssdlite_exporter.py (行20-52)
class _SSDLiteExportWrapper(nn.Module):
    def __init__(self, ssd_model: nn.Module) -> None:
        self.backbone = ssd_model.backbone  # backbone のみ
        self.head = ssd_model.head          # head のみ
        # transform は含まれない ← GeneralizedRCNNTransform 欠落

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        features_list = list(features.values())
        head_out = self.head(features_list)
        return head_out["cls_logits"], head_out["bbox_regression"]
```

---

## 修正方針

1. `SsdCocoDataset` と `infer.py` から ImageNet 正規化を除去 (不要な正規化の排除)
2. `_SSDLiteExportWrapper.forward()` に `GeneralizedRCNNTransform` の normalize を組み込み (ONNX モデルに正規化を焼き込み)
3. 再学習が必要 (既存モデルは二重正規化で学習済み)

---

## 結論

| 項目 | 説明 |
|------|------|
| 根本原因 | Dataset/Pipeline の ImageNet 正規化が不要, かつ ONNX パスで `GeneralizedRCNNTransform` が欠落 |
| 影響範囲 | SSDLite の学習・推論パス全体 (RT-DETR は影響なし) |
| 修正コスト | コード変更は小規模, ただし再学習が必要 |
| 期待される効果 | ONNX と PyTorch の mAP が同等になり, さらに正しい入力分布での学習により mAP 改善が期待できる |
