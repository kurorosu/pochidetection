# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- `export-trt` コマンドが SSDLite に対応. SSDLite ONNX モデルから TensorRT エンジン (FP32/FP16) をビルド可能にした. ([#233](https://github.com/kurorosu/pochidetection/pull/233).)

### Changed
- `RTDetrTensorRTExporter` を `TensorRTExporter` にリネームし, `tensorrt/rtdetr/` から `tensorrt/` へ昇格. アーキテクチャ非依存の実態に合わせた. ([#234](https://github.com/kurorosu/pochidetection/pull/234).)
- RT-DETR / SSDLite 個別の TRT エクスポートスクリプトを `scripts/common/export_trt.py` に統合. ([#234](https://github.com/kurorosu/pochidetection/pull/234).)

### Fixed
- 無し.

### Removed
- 無し.

## v0.8.0 (2026-03-10)

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
- SSDLite の二重正規化バグを修正. Dataset/Pipeline の ImageNet 正規化を除去し, `_SSDLiteExportWrapper` に `GeneralizedRCNNTransform` 相当の正規化を組み込み. PyTorch と ONNX の入力分布を統一. ([#227](https://github.com/kurorosu/pochidetection/pull/227).)
- SSDLite 推論パイプラインのリサイズ補間を `PIL.Image.resize` (bicubic) から `v2.Resize` (bilinear) に統一. 学習時と推論時の前処理を一致させ, mAP 約 3% の乖離を解消. ([#229](https://github.com/kurorosu/pochidetection/pull/229).)

### Removed
- 無し.

## Archived Changelogs

- [0.7.x](changelogs/0.7.x.md)
- [0.6.x](changelogs/0.6.x.md)
- [0.5.x](changelogs/0.5.x.md)
- [0.4.x](changelogs/0.4.x.md)
- [0.3.x](changelogs/0.3.x.md)
- [0.2.x](changelogs/0.2.x.md)
- [0.1.x](changelogs/0.1.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
