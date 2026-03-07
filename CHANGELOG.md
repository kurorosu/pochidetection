# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- README に SSDLite の使い方, Early Stopping, LR Scheduler の説明を追加. torchvision バッジを追加. ([#161](https://github.com/kurorosu/pochidetection/pull/161).)

### Changed
- SSDLite / RT-DETR 間の重複コード解消. (N/A.)
  - `BaseCocoDataset` 基底クラスを新設し, `CocoDetectionDataset` / `SsdCocoDataset` の共通ロジック (アノテーション読み込み, カテゴリ管理, 画像ロード) を集約.
  - `scripts/common/training.py` に学習ループ共通ロジック (`run_training_loop`, `train_one_epoch`, `build_early_stopping`, `save_results`) を抽出し, `rtdetr/train.py` / `ssdlite/train.py` から重複コードを除去.
  - `scripts/common/inference.py` に推論共通ロジック (`resolve_model_path`, `collect_image_files`, `write_reports`) を抽出し, `rtdetr/infer.py` / `ssdlite/infer.py` から重複コードを除去.
- `IDetectionModel.forward()` の返り値と labels キー名を統一. (N/A.)
  - `IDetectionModel.forward()` の返り値契約を明確化 (`loss`: 学習時必須, `predictions`: 推論時必須).
  - `SsdCocoDataset` のラベルキーを `"labels"` → `"class_labels"` に変更し, `CocoDetectionDataset` と統一.
  - `SSDLiteModel.forward()` が `class_labels` キーを受け取るよう変更.
- `IDetectionModel` に `save()` / `load()` 抽象メソッドを追加し, モデル永続化の契約を定義. (N/A.)
  - `RTDetrModel` に processor を内包させ, `save()` / `load()` でモデルと processor を一括保存・復元.
  - `SSDLiteModel` の `__init__` から `model_path` を除去し, `save()` / `load()` で state_dict を保存・復元.
  - `train.py` の `_save_model` 関数と `ModelSaver` Protocol を削除し, `ctx.model.save()` に統一.

### Fixed
- 無し.

### Removed
- 無し.

## v0.6.0 (2026-03-07)

### Added
- `DetectionConfig` に `early_stopping_patience` / `early_stopping_metric` / `early_stopping_min_delta` フィールドを追加し, 学習時の Early Stopping を使用可能にした. ([#150](https://github.com/kurorosu/pochidetection/pull/150).)
- SSDLite MobileNetV3 アーキテクチャを追加. `SSDLiteModel` (torchvision ラッパー), `SsdCocoDataset` (xyxy + 1-indexed ラベル), 学習・推論パイプライン, `configs/ssdlite_coco.py` を新設. config の `architecture = "SSDLite"` で切り替え可能. ([#151](https://github.com/kurorosu/pochidetection/pull/151).)

### Changed
- `pochidetection/cli/rtdetr.py` を `pochidetection/cli/main.py` にリネームし, アーキテクチャ非依存の CLI エントリーポイントに変更. ([#151](https://github.com/kurorosu/pochidetection/pull/151).)

### Fixed
- 無し.

### Removed
- 無し.

## Archived Changelogs

- [0.5.x](changelogs/0.5.x.md)

- [0.4.x](changelogs/0.4.x.md)
- [0.3.x](changelogs/0.3.x.md)
- [0.2.x](changelogs/0.2.x.md)
- [0.1.x](changelogs/0.1.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
