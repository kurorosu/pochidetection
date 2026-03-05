# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- `DetectionConfig` に `lr_scheduler` / `lr_scheduler_params` フィールドを追加し, 学習時に PyTorch 標準の Learning Rate Scheduler を使用可能にした. `CosineAnnealingLR` は `T_max` 未指定時にエポック数をデフォルトとする. ([#134](https://github.com/kurorosu/pochidetection/pull/134).)
- `pochi infer` 実行後に画像ごと・検出ごとの推論結果を `detection_results.csv` として出力する機能を追加. アノテーション指定時は TP/FP/FN 正答状況と IoU を含む. ([#135](https://github.com/kurorosu/pochidetection/pull/135).)
- `pochi infer` でアノテーション指定時に `confusion_matrix.html` (Plotly ヒートマップ) を出力する機能を追加. Background 行/列で FP/FN を表現. ([#136](https://github.com/kurorosu/pochidetection/pull/136).)

### Changed
- `LoggerManager` の `typing.Optional` を `X | None` 記法に統一. (N/A.)
- `EpochMetrics` を内部専用クラスとして `_EpochMetrics` にリネーム. (N/A.)

### Fixed
- `Visualizer` のラベルテキストが常に白で描画され, 黄色等の明るい背景色で視認性が低下する問題を修正. W3C 相対輝度に基づいて黒/白を自動切替するよう変更. ([#133](https://github.com/kurorosu/pochidetection/pull/133).)

### Removed
- プロダクション未使用の `LoggerManager.get_available_loggers()` メソッドを削除. (N/A.)
- プロダクション未使用の `WorkspaceManager.get_training_state_path()` メソッドおよび専用テストを削除. (N/A.)

## v0.4.4 (2026-03-05)

### Added
- `pochi infer` 実行後にクラス毎の検出数・平均スコア・検出画像数をサマリーログ出力し, `detection_summary.json` として保存する機能を追加. ([#112](https://github.com/kurorosu/pochidetection/pull/112).)

### Changed
- `CocoDetectionDataset` と `MapEvaluator` のカテゴリフィルタリング・ID→index マッピングロジックを `category_utils` に共通化. ([#121](https://github.com/kurorosu/pochidetection/pull/121).)
- `PRCurvePlotter` と `TrainingReportPlotter` の `LEGEND_CONFIG` 定数を `plotters/constants.py` に共通化. ([#122](https://github.com/kurorosu/pochidetection/pull/122).)
- `LossPlotter` と `MetricsPlotter` の重複 `plot()` メソッドを `IPlotter` のデフォルト実装に集約. ([#123](https://github.com/kurorosu/pochidetection/pull/123).)
- `WorkspaceManager` の4メソッドに重複していたワークスペース未作成チェックを `_ensure_workspace_created()` に集約. ([#124](https://github.com/kurorosu/pochidetection/pull/124).)
- `DetectionPipeline` と `infer.py` に分散していた FP16 判定ロジックを `is_fp16_available()` に共通化. ([#125](https://github.com/kurorosu/pochidetection/pull/125).)

### Fixed
- なし.

### Removed
- `DetectionPipeline` 導入後に未使用となっていた `Detector` クラスおよび専用テスト `test_detector.py` を削除. ([#120](https://github.com/kurorosu/pochidetection/pull/120).)

## Archived Changelogs

- [0.4.x](changelogs/0.4.x.md)
- [0.3.x](changelogs/0.3.x.md)
- [0.2.x](changelogs/0.2.x.md)
- [0.1.x](changelogs/0.1.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
