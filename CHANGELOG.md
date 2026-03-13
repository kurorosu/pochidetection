# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- `F1ConfidencePlotter` を追加. 学習時に信頼度閾値ごとの F1 スコア変化を `f1_confidence.html` として可視化. ([#251](https://github.com/kurorosu/pochidetection/pull/251).)

### Changed
- `cli/main.py` を `parser.py` + `commands/` に分割. main.py を 50 行以下に縮小. (N/A.)

### Removed
- 無し.

### Fixed
- 無し.

## v0.9.0 (2026-03-12)

### Added
- `export-trt` コマンドが SSDLite に対応. SSDLite ONNX モデルから TensorRT エンジン (FP32/FP16) をビルド可能にした. ([#233](https://github.com/kurorosu/pochidetection/pull/233).)
- `SSDLiteTensorRTBackend` を追加. `pochi infer -m model.engine` で SSDLite TensorRT エンジンの推論に対応. ([#236](https://github.com/kurorosu/pochidetection/pull/236).)
- `INT8Calibrator` を追加. TensorRT INT8 Post-Training Quantization (PTQ) に対応. CLI `--int8` フラグで INT8 エンジンをビルド可能. キャリブレーション画像は config の `infer_image_dir` から取得. ([#242](https://github.com/kurorosu/pochidetection/pull/242).)

### Changed
- `RTDetrTensorRTExporter` を `TensorRTExporter` にリネームし, `tensorrt/rtdetr/` から `tensorrt/` へ昇格. アーキテクチャ非依存の実態に合わせた. ([#234](https://github.com/kurorosu/pochidetection/pull/234).)
- RT-DETR / SSDLite 個別の TRT エクスポートスクリプトを `scripts/common/export_trt.py` に統合. ([#234](https://github.com/kurorosu/pochidetection/pull/234).)
- `RTDetrPipeline` の前処理を HuggingFace `RTDetrImageProcessor` から torchvision v2 transforms に置換. 前処理時間を ~9ms → ~1.7ms に短縮. ([#239](https://github.com/kurorosu/pochidetection/pull/239).)
- `export` と `export-trt` コマンドを `export` に統合. `-m` にフォルダを指定すると ONNX エクスポート, `.onnx` ファイルを指定すると TensorRT エクスポートを自動実行. ([#240](https://github.com/kurorosu/pochidetection/pull/240).)

### Removed
- `export-trt` コマンドを削除. `export -m model.onnx` に統合. ([#240](https://github.com/kurorosu/pochidetection/pull/240).)

### Fixed
- 無し.

## Archived Changelogs

- [0.8.x](changelogs/0.8.x.md)
- [0.7.x](changelogs/0.7.x.md)
- [0.6.x](changelogs/0.6.x.md)
- [0.5.x](changelogs/0.5.x.md)
- [0.4.x](changelogs/0.4.x.md)
- [0.3.x](changelogs/0.3.x.md)
- [0.2.x](changelogs/0.2.x.md)
- [0.1.x](changelogs/0.1.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
