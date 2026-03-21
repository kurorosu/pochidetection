# Data Augmentation 設定ガイド

## 概要

pochidetection は `torchvision.transforms.v2` を使用した Data Augmentation パイプラインを提供する. 学習時のみ適用され, 検証・推論時には適用されない. バウンディングボックスは画像と同時に変換される.

## 基本設定

config.py の `augmentation` セクションで設定する.

```python
# configs/rtdetr_coco.py
augmentation = {
    "enabled": True,  # False で無効化
    "transforms": [
        {"name": "RandomHorizontalFlip", "p": 0.5},
        {"name": "ColorJitter", "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
    ],
}
```

- `enabled`: `True` で有効, `False` で全変換を無効化
- `transforms`: 適用する変換のリスト. 上から順に適用される
- `name`: `torchvision.transforms.v2` のクラス名
- `p`: 適用確率 (0.0-1.0, デフォルト: 1.0)
- その他のキーはそのまま変換クラスのコンストラクタ引数として渡される

## augmentation を無効化する方法

```python
# 方法 1: enabled を False にする
augmentation = {
    "enabled": False,
    "transforms": [...],
}

# 方法 2: augmentation 自体を指定しない (デフォルト: None)
# augmentation = None  # または行ごと削除
```

## 使用可能な変換一覧

### 幾何変換 (bbox も自動変換)

| 変換名 | 主なパラメータ | 説明 |
|--------|-------------|------|
| `RandomHorizontalFlip` | `p` | 水平反転 |
| `RandomVerticalFlip` | `p` | 垂直反転 |
| `RandomRotation` | `degrees` | 回転. `degrees=30` で -30~+30 度 |
| `RandomAffine` | `degrees`, `translate`, `scale`, `shear` | アフィン変換 |
| `RandomPerspective` | `distortion_scale`, `p` | 透視変換 |
| `RandomResizedCrop` | `size`, `scale`, `ratio` | ランダムクロップ + リサイズ |
| `RandomZoomOut` | `fill`, `side_range`, `p` | ランダムズームアウト (SSD の expand 相当) |

### 色変換 (bbox に影響なし)

| 変換名 | 主なパラメータ | 説明 |
|--------|-------------|------|
| `ColorJitter` | `brightness`, `contrast`, `saturation`, `hue` | 明度・コントラスト・彩度・色相をランダム変更 |
| `GaussianBlur` | `kernel_size`, `sigma` | ガウシアンぼかし |
| `RandomGrayscale` | `p` | ランダムにグレースケール化 |
| `RandomAutocontrast` | - | 自動コントラスト調整 |
| `RandomEqualize` | - | ヒストグラム均一化 |
| `RandomPosterize` | `bits` | ポスタライズ |
| `RandomSolarize` | `threshold` | ソラリゼーション |
| `RandomPhotometricDistort` | `brightness`, `contrast`, `saturation`, `hue`, `p` | SSD 論文の色変換をまとめたもの |

### 特殊

| 変換名 | 主なパラメータ | 説明 |
|--------|-------------|------|
| `RandomErasing` | `p`, `scale`, `ratio` | ランダムに矩形領域を消去 (Cutout 相当) |

## 適用確率 (p) の扱い

変換には 2 種類ある:

1. **自前で `p` を持つ変換** (`RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomPhotometricDistort`)
   - `p` をそのまま変換クラスに渡す

2. **`p` を持たない変換** (`ColorJitter`, `GaussianBlur` 等)
   - `p < 1.0` の場合, `RandomApply` でラップして確率的に適用
   - `p = 1.0` (デフォルト) の場合, 常に適用

## 設定例

### 軽量な拡張 (推奨: 小規模データセット)

```python
augmentation = {
    "enabled": True,
    "transforms": [
        {"name": "RandomHorizontalFlip", "p": 0.5},
        {"name": "ColorJitter", "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
    ],
}
```

### 強めの拡張 (データセットが非常に小さい場合)

```python
augmentation = {
    "enabled": True,
    "transforms": [
        {"name": "RandomHorizontalFlip", "p": 0.5},
        {"name": "RandomPhotometricDistort", "p": 0.5},
        {"name": "RandomZoomOut", "p": 0.3, "side_range": [1.0, 2.0]},
        {"name": "RandomRotation", "degrees": 15},
        {"name": "GaussianBlur", "kernel_size": 3, "p": 0.1},
        {"name": "RandomErasing", "p": 0.1, "scale": [0.02, 0.1]},
    ],
}
```

### 色変換のみ (bbox に影響を与えたくない場合)

```python
augmentation = {
    "enabled": True,
    "transforms": [
        {"name": "ColorJitter", "brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.15},
        {"name": "GaussianBlur", "kernel_size": 5, "p": 0.2},
    ],
}
```

## 注意事項

- Augmentation は **学習データのみ** に適用される. 検証・推論データには適用されない
- 幾何変換は bbox を同時に変換する (`torchvision.tv_tensors.BoundingBoxes` を使用)
- 変換後に面積ゼロになった bbox は自動的に除外される
- `Mosaic`, `MixUp` は `torchvision.transforms.v2` に含まれないため使用不可 (必要な場合は `albumentations` の導入を検討)
- 変換名は `torchvision.transforms.v2` のクラス名と完全一致する必要がある (不明な名前は警告スキップ)
