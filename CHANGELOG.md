# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- `DetectionConfig` に `infer_image_dir` フィールドを追加. CLI `-d` 未指定時に config から推論対象の画像フォルダパスを解決できるようにした. ([#219](https://github.com/kurorosu/pochidetection/pull/219).)
- `SSDLiteOnnxExporter` を追加. SSDLite モデルの ONNX エクスポート (FP32/FP16) に対応. CLI `export` コマンドで SSDLite config 使用時に自動ディスパッチ. ([#221](https://github.com/kurorosu/pochidetection/pull/221).)
- `SSDLiteOnnxBackend` を追加. SSDLite ONNX モデルの推論バックエンド (アンカーデコード + NMS 後処理). `SSDLitePyTorchBackend` と合わせ, `SSDLitePipeline` を backend ベースの Strategy パターンに変更. ([#222](https://github.com/kurorosu/pochidetection/pull/222).)

### Changed
- `SSDLiteModel` が `nms_iou_threshold` を受け取り, torchvision の `nms_thresh` に渡すように変更. 設定ファイルで NMS IoU 閾値を制御可能にした. ([#220](https://github.com/kurorosu/pochidetection/pull/220).)
- `IInferenceBackend.infer()` の戻り値型を `tuple[Any, Any]` から `Any` に変更. SSDLite バックエンドが `dict[str, Tensor]` を返せるようにした. ([#222](https://github.com/kurorosu/pochidetection/pull/222).)
- `SSDLitePipeline` のコンストラクタを `model: SSDLiteModel` から `backend: IInferenceBackend` に変更. ([#222](https://github.com/kurorosu/pochidetection/pull/222).)
- RT-DETR 系モジュールの命名統一とフォルダ構成整理. `OnnxBackend` → `RTDetrOnnxBackend`, `PyTorchBackend` → `RTDetrPyTorchBackend`, `TensorRTBackend` → `RTDetrTensorRTBackend`, `OnnxExporter` → `RTDetrOnnxExporter`, `TensorRTExporter` → `RTDetrTensorRTExporter`, `DetectionPipeline` → `RTDetrPipeline` にリネーム. `inference/` と `tensorrt/` をモデル別サブフォルダに分割. ([#226](https://github.com/kurorosu/pochidetection/pull/226).)

### Fixed
- SSDLite の二重正規化バグを修正. Dataset/Pipeline の ImageNet 正規化を除去し, `_SSDLiteExportWrapper` に `GeneralizedRCNNTransform` 相当の正規化を組み込み. PyTorch と ONNX の入力分布を統一. (N/A.)
- SSDLite 推論パイプラインのリサイズ補間を `PIL.Image.resize` (bicubic) から `v2.Resize` (bilinear) に統一. 学習時と推論時の前処理を一致させ, mAP 約 3% の乖離を解消. (N/A.)

### Removed
- 無し.

## v0.7.0 (2026-03-08)

### Added
- 無し.

### Changed
- CLI の description を汎用化し, ヘルプに SSDLite の使用例と export の RT-DETR 限定注記を追加. `export` / `export-trt` コマンドに SSDLite 設定ファイル指定時のガードを追加. ([#209](https://github.com/kurorosu/pochidetection/pull/209).)
- `DetectionConfig.architecture` に `field_validator(mode="before")` を追加し, 大文字小文字を問わず `"RTDetr"` / `"SSDLite"` に正規化. ([#210](https://github.com/kurorosu/pochidetection/pull/210).)
- `WorkspaceManager.save_config()` が元のファイル名を保持してコピーするように変更. `resolve_config_path()` が `config.py` 以外の `.py` ファイルも自動検出するように拡張. ([#211](https://github.com/kurorosu/pochidetection/pull/211).)
- `TensorRTExporter.export()` のメモリプール制限を `build_memory` パラメータとして外部化. CLI に `--build-memory` オプションを追加. ([#212](https://github.com/kurorosu/pochidetection/pull/212).)

### Fixed
- 無し.

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
