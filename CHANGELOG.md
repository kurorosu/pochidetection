# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し.

### Changed
- 無し.

### Fixed
- 無し.

### Removed
- 無し.

## v0.6.2 (2026-03-07)

### Added
- 無し.

### Changed
- 無し.

### Fixed
- `DetectionConfig` で SSDLite が無視する設定項目 (`model_name`, `pretrained`, `nms_iou_threshold`) にデフォルト以外の値を指定した場合に `UserWarning` を発行するバリデーションを追加. サンプル設定 `ssdlite_coco.py` から該当項目を削除. ([#168](https://github.com/kurorosu/pochidetection/pull/168).)
- SSDLite の label オフセット (+1/-1) を `SSDLiteModel.forward()` に集約し, `SsdCocoDataset` と `train.py _validate()` から分散していたオフセット処理を除去. 推論時の背景クラス予測 (label=-1) を除去するガードを追加. ([#169](https://github.com/kurorosu/pochidetection/pull/169).)
- SSDLite の前処理に `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` を追加. `SsdCocoDataset` (学習) と `ssdlite/infer.py` (推論) の両方に適用し, MobileNetV3 バックボーンが期待する ImageNet 正規化を実施. ([#171](https://github.com/kurorosu/pochidetection/pull/171).)

### Removed
- 無し.

## Archived Changelogs

- [0.6.x](changelogs/0.6.x.md)
- [0.5.x](changelogs/0.5.x.md)
- [0.4.x](changelogs/0.4.x.md)
- [0.3.x](changelogs/0.3.x.md)
- [0.2.x](changelogs/0.2.x.md)
- [0.1.x](changelogs/0.1.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
