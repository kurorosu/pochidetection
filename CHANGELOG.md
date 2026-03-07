# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- `DetectionConfig` に `early_stopping_patience` / `early_stopping_metric` / `early_stopping_min_delta` フィールドを追加し, 学習時の Early Stopping を使用可能にした. (N/A.)

### Changed
- 無し.

### Fixed
- 無し.

### Removed
- 無し.

## v0.5.0 (2026-03-06)

### Added
- `DetectionConfig` に `lr_scheduler` / `lr_scheduler_params` フィールドを追加し, 学習時に PyTorch 標準の Learning Rate Scheduler を使用可能にした. `CosineAnnealingLR` は `T_max` 未指定時にエポック数をデフォルトとする. ([#134](https://github.com/kurorosu/pochidetection/pull/134).)
- `pochi infer` 実行後に画像ごと・検出ごとの推論結果を `detection_results.csv` として出力する機能を追加. アノテーション指定時は TP/FP/FN 正答状況と IoU を含む. ([#135](https://github.com/kurorosu/pochidetection/pull/135).)
- `pochi infer` でアノテーション指定時に `confusion_matrix.html` (Plotly ヒートマップ) を出力する機能を追加. Background 行/列で FP/FN を表現. ([#136](https://github.com/kurorosu/pochidetection/pull/136).)

### Changed
- テスト品質改善. (N/A.)
  - `PyTorchBackend` の `infer` / `synchronize` テストを追加
  - `pytest.importorskip("onnx")` をルート conftest から `tests/test_onnx/conftest.py` に移動
  - `test_onnx_backend.py` を `test_inference/` から `test_onnx/` に移動
  - `confusion_matrix_plotter` / `detection_results_writer` の `>= N` アサーションを固定データに基づく厳密値に変更
  - `timer` テストに上限チェック (`< 500.0` ms) を追加
  - `coco_annotation` / `sample_predictions` fixture をルート conftest に共通化
  - `test_scheduler.py` の `scheduler.step()` 前に `optimizer.step()` を追加し, PyTorch UserWarning を解消
- `test_timer` の CUDA フォールバックテストに上限チェック (`< 500.0` ms) を追加し, 片側アサーションを解消. (N/A.)
- デッドコード・未使用コードの削除. ([#142](https://github.com/kurorosu/pochidetection/pull/142).)
  - `LoggerManager` の `typing.Optional` を `X | None` 記法に統一
  - `EpochMetrics` を内部専用クラス `_EpochMetrics` にリネーム
- コード品質改善 (assert 修正・型アノテーション・マジックナンバー・docstring). ([#143](https://github.com/kurorosu/pochidetection/pull/143).)
  - `infer.py` の `assert` を `TypeError` 例外に変更
  - `dict` 型アノテーションに値型を明示 (`map_evaluator`, `confusion_matrix_plotter`, `detection_results_writer`)
  - `visualizer.py` のマジックナンバーをクラス定数に抽出
  - `benchmark.py` の docstring タイポを修正
  - `history.py` の property docstring に `Returns:` セクションを追加
- COCO GT ローダー共通化と重複コード解消. ([#144](https://github.com/kurorosu/pochidetection/pull/144).)
  - `extract_basename`, `xywh_to_xyxy` を `coco_utils.py` の public 関数に移動
  - `load_coco_ground_truth` で JSON 読み込み・マッピング構築・GT グルーピングを共通化
  - `MapEvaluator`, `confusion_matrix_plotter`, `detection_results_writer` から重複ロジックを除去
- 設計・アーキテクチャ改善. ([#145](https://github.com/kurorosu/pochidetection/pull/145).)
  - `Detection` / `OutputWrapper` を `scripts/rtdetr/inference/` から `core/` に移動し, `visualization → scripts` のレイヤー違反を解消
  - `infer()` を `_setup_pipeline()` / `_run_inference()` / `_write_reports()` に分割
  - `train()` を `_setup_training()` / `_build_data_loaders()` / `_train_one_epoch()` / `_validate()` / `_save_results()` に分割
  - `IPlotter` を `ITrainingCurvePlotter` にリネームし, `IReportPlotter` を新設
  - `PRCurvePlotter` / `ConfusionMatrixPlotter` / `TrainingReportPlotter` を `IReportPlotter` に準拠

### Fixed
- `Visualizer` のラベルテキストが常に白で描画され, 黄色等の明るい背景色で視認性が低下する問題を修正. W3C 相対輝度に基づいて黒/白を自動切替するよう変更. ([#133](https://github.com/kurorosu/pochidetection/pull/133).)

### Removed
- デッドコード・未使用コードの削除. ([#142](https://github.com/kurorosu/pochidetection/pull/142).)
  - プロダクション未使用の `LoggerManager.get_available_loggers()` メソッドを削除
  - プロダクション未使用の `WorkspaceManager.get_training_state_path()` メソッドおよび専用テストを削除

## Archived Changelogs

- [0.4.x](changelogs/0.4.x.md)
- [0.3.x](changelogs/0.3.x.md)
- [0.2.x](changelogs/0.2.x.md)
- [0.1.x](changelogs/0.1.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
