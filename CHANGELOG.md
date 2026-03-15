# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## v0.11.0 (2026-03-15)

### Added
- `ConfigLoader.load()`, `load_coco_ground_truth()`, `MapEvaluator.evaluate()` に Examples セクションを追加. ([#341](https://github.com/kurorosu/pochidetection/pull/341).)
- TensorBoard 統合による学習メトリクスのリアルタイムモニタリング機能を追加. `enable_tensorboard = True` で Loss, mAP, 学習率をエポック単位で記録. ([#347](https://github.com/kurorosu/pochidetection/pull/347).)

### Changed
- 学習時に config の `cudnn_benchmark` 設定を反映するよう `setup_training()` に `setup_cudnn_benchmark()` 呼び出しを追加. ([#342](https://github.com/kurorosu/pochidetection/pull/342).)
- サブコマンド未指定時に argparse の `required=True` でエラーメッセージと非ゼロ終了コードを返すよう変更. フォールバックのヘルプ表示ロジックを削除. ([#339](https://github.com/kurorosu/pochidetection/pull/339).)
- mAP 関連のプロパティ名・フィールド名・CSV カラム名を snake_case に統一 (`mAP` → `map`, `mAP_50` → `map_50`, `mAP_75` → `map_75`). ([#340](https://github.com/kurorosu/pochidetection/pull/340).)

### Removed
- `tests/conftest.py` の未使用 `training_history` fixture を削除. ([#349](https://github.com/kurorosu/pochidetection/pull/349).)
- `tests/test_tensorrt/conftest.py` の未使用 `int8_engine_path` fixture を削除. ([#349](https://github.com/kurorosu/pochidetection/pull/349).)
- `DetectionConfig` / `DetectionConfigDict` から未使用フィールド `loss`, `metrics`, `dataset` を削除. ([#350](https://github.com/kurorosu/pochidetection/pull/350).)

### Fixed
- SSDLite 推論時の `nms_iou_threshold` デフォルト値を `0.55` から `0.5` に修正し, スキーマ・学習側と統一. ([#348](https://github.com/kurorosu/pochidetection/pull/348).)

## Archived Changelogs

- [0.10.x](changelogs/0.10.x.md)
- [0.9.x](changelogs/0.9.x.md)
- [0.8.x](changelogs/0.8.x.md)
- [0.7.x](changelogs/0.7.x.md)
- [0.6.x](changelogs/0.6.x.md)
- [0.5.x](changelogs/0.5.x.md)
- [0.4.x](changelogs/0.4.x.md)
- [0.3.x](changelogs/0.3.x.md)
- [0.2.x](changelogs/0.2.x.md)
- [0.1.x](changelogs/0.1.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
