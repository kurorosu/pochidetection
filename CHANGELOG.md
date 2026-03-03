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
- `IDetectionModel` / `RTDetrModel` の `get_backbone_params()` / `get_head_params()` を削除. 未使用であり, differential learning rate は未実装のため. (N/A.)

## v0.4.1 (2026-03-03)

### Added
- なし.

### Changed
- なし.

### Fixed
- `MapEvaluator.evaluate()` が predictions 側のみ走査していたため, GT に存在するが推論されなかった画像が評価から除外され mAP が過大評価される問題を修正. GT 全画像を起点に走査するよう変更. ([#85](https://github.com/kurorosu/pochidetection/pull/85).)
- `PRCurvePlotter._create_per_class_figure()` で precision の無効値 (-1) が NaN に置換されず, per-class グラフにそのまま描画される問題を修正. ([#86](https://github.com/kurorosu/pochidetection/pull/86).)
- `build_benchmark_result()` の `avg_e2e_ms` 計算を「各フェーズの average_ms の合計」から「全フェーズの total_ms 合計 / count」に修正. フェーズ間で count が異なる場合に正しい per-image E2E 時間を算出するよう変更. ([#87](https://github.com/kurorosu/pochidetection/pull/87).)
- `LabelMapper.get_label()` が負の `class_id` で Python の負のインデックスにより誤ったクラス名を返す問題を修正. `0 <=` の下限チェックを追加. ([#88](https://github.com/kurorosu/pochidetection/pull/88).)
- `pochi export` / `pochi export-trt` で `-c` 省略時にデフォルト config を使用していた問題を修正. `infer` と同じ `resolve_config_path` でモデルディレクトリから config.py を自動解決するよう変更. ([#89](https://github.com/kurorosu/pochidetection/pull/89).)
- `InferenceSaver._create_numbered_dir()` の glob パターンが 3 桁固定で 4 桁以上のディレクトリを検出できず, 1000 回目以降にデータが上書きされる問題を修正. 正規表現による任意桁数マッチに変更し, `exist_ok=True` を削除. ([#90](https://github.com/kurorosu/pochidetection/pull/90).)
- `train()` で DataLoader が空の場合に `ZeroDivisionError` が発生する問題を修正. 学習ループ前にバリデーションを追加し, 分かりやすいエラーメッセージで早期終了するよう変更. ([#91](https://github.com/kurorosu/pochidetection/pull/91).)
- `CocoDetectionDataset` と `MapEvaluator` のカテゴリ ID→インデックス変換が JSON 内の出現順に依存していた問題を修正. カテゴリ ID の昇順でソートしてから連続インデックスを割り当てるよう変更. ([#92](https://github.com/kurorosu/pochidetection/pull/92).)

### Removed
- なし.

## Archived Changelogs

- [0.4.x](changelogs/0.4.x.md)
- [0.3.x](changelogs/0.3.x.md)
- [0.2.x](changelogs/0.2.x.md)
- [0.1.x](changelogs/0.1.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
