# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- ONNX Runtime 推論バックエンド `OnnxBackend` を追加. `-m model.onnx` で自動判定し, `config["device"]` に連動した Execution Providers 選択と実デバイスのベンチマーク反映に対応. ([#53](https://github.com/kurorosu/pochidetection/pull/53).)
- TensorRT (FP32) エクスポート機能 (`pochidet-rtdetr export-trt`) およびエクスポートクラス (`TensorRTExporter`) を追加. Dynamic Batching 設定に対応. ([#61](https://github.com/kurorosu/pochidetection/pull/61).)
- TensorRT 推論バックエンド `TensorRTBackend` を追加. `-m model.engine` で自動判定し, `execute_async_v3` + 非デフォルト CUDA ストリームで非同期推論を実行. 動的バッチサイズエンジンに対応し, `PhasedTimer` によるベンチマーク計測と `benchmark_result.json` 出力に対応. ([#63](https://github.com/kurorosu/pochidetection/pull/63).)
- 推論後処理にクラス非依存 NMS (`torchvision.ops.nms`) をデフォルト有効 (IoU=0.5) で追加. `--nms-iou` CLI オプションで閾値を変更可能. ([#67](https://github.com/kurorosu/pochidetection/pull/67).)
- 推論ベンチマークに mAP 精度評価 (`MapEvaluator`) を追加. config の `annotation_path` に COCO アノテーションを指定すると mAP@0.5 / mAP@0.5:0.95 を自動計算し `benchmark_result.json` に含める. ([#67](https://github.com/kurorosu/pochidetection/pull/67).)
- 推論時にワークスペースの `config.py` を自動解決する機能を追加. `-m` で指定したモデルの親ディレクトリから `config.py` を検出し, `-c` 省略時に自動適用. ([#69](https://github.com/kurorosu/pochidetection/pull/69).)
- `DetectionConfig` に `train_score_threshold`, `infer_score_threshold`, `nms_iou_threshold` フィールドを追加. ([#69](https://github.com/kurorosu/pochidetection/pull/69).)
- `TensorRTExporter` に FP16 エクスポート機能を追加. `--fp16` CLI オプションで FP16 Mixed Precision エンジンをビルド可能. FP16 非対応 GPU では警告ログを出力し FP32 にフォールバック. (N/A.)

### Changed
- ONNX / TensorRT バックエンドの出力テンソル取得をインデックスベースから名前ベースに変更. エクスポート設定変更時の出力順依存を排除. ([#67](https://github.com/kurorosu/pochidetection/pull/67).)
- `pytest-xdist` を導入しテストを 6 ワーカーで並列実行するよう変更. 重複していた `onnx_path` fixture をルート `conftest.py` に統合. ([#68](https://github.com/kurorosu/pochidetection/pull/68).)
- 推論時の閾値を CLI 引数 (`-t`, `--nms-iou`) から config.py ベースに変更. CLI オプションは廃止. ([#69](https://github.com/kurorosu/pochidetection/pull/69).)
- 学習時の mAP 計算閾値をハードコーディング (0.2) から `train_score_threshold` config フィールドに変更. ([#69](https://github.com/kurorosu/pochidetection/pull/69).)

### Fixed
- `DetectionConfig.train_score_threshold` のデフォルト値を `0.2` から `0.5` に修正. ([#70](https://github.com/kurorosu/pochidetection/pull/70).)

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
