# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- なし.

### Changed
- なし.

### Fixed
- なし.

### Removed
- なし.

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
