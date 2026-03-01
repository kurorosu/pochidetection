# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- ONNX Runtime 推論バックエンド `OnnxBackend` を追加. `-m model.onnx` で自動判定し, `config["device"]` に連動した Execution Providers 選択と実デバイスのベンチマーク反映に対応. ([#53](https://github.com/kurorosu/pochidetection/pull/53).)
- TensorRT (FP32) エクスポート機能 (`pochidet-rtdetr export-trt`) およびエクスポートクラス (`TensorRTExporter`) を追加. Dynamic Batching 設定に対応. ([#61](https://github.com/kurorosu/pochidetection/pull/61).)
- TensorRT 推論バックエンド `TensorRTBackend` を追加. `-m model.engine` で自動判定し, `execute_async_v3` + 非デフォルト CUDA ストリームで非同期推論を実行. 動的バッチサイズエンジンに対応し, `PhasedTimer` によるベンチマーク計測と `benchmark_result.json` 出力に対応. (N/A.)

### Changed
- なし.

### Fixed
- なし.

### Removed
- なし.

## v0.3.0 (2026-03-01)

### Added
- ベンチマーク結果スキーマ `BenchmarkResult` (Pydantic) と JSON 出力機能 (`build_benchmark_result`, `write_benchmark_result`) を追加. 推論完了時に `benchmark_result.json` を出力する. ([#48](https://github.com/kurorosu/pochidetection/pull/48))

### Changed
- `infer` コマンドを `DetectionPipeline` + `PhasedTimer` ベースに移行し, フェーズ別タイミング・スループットをターミナル出力するよう変更. ([#48](https://github.com/kurorosu/pochidetection/pull/48))
- `Detector` から `timer` パラメータを削除し, 推論計測責務を `infer` 側 (`DetectionPipeline` + `PhasedTimer`) に統一. ([#49](https://github.com/kurorosu/pochidetection/pull/49))
- 低価値コメントを整理し, `TODO` と実装意図を示すコメントのみを保持して可読性を改善. ([#50](https://github.com/kurorosu/pochidetection/pull/50))

### Fixed
- なし.

### Removed
- なし.

## Archived Changelogs

- [0.2.x](changelogs/0.2.x.md)
- [0.1.x](changelogs/0.1.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
