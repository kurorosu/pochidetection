# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

無し

## v0.14.0 (2026-03-21)

### Added
- 画像推論で検出ボックスのクロップ画像を `inference_XXX/crop/` に自動保存する機能を追加 (デフォルト有効). `--no-crop` で無効化可能. ([#396](https://github.com/kurorosu/pochidetection/pull/396))
  - `InferFn` の型定義を `Callable` から `Protocol` に変更し, keyword-only 引数 (`save_crop`) に対応.
- `torchvision.transforms.v2` ベースの Data Augmentation パイプラインを導入. config.py の `augmentation` セクションで変換を設定可能. 学習データのみに適用, bbox は `tv_tensors.BoundingBoxes` で同時変換. ([#397](https://github.com/kurorosu/pochidetection/pull/397))
  - `docs/augmentation.md` に設定方法・変換一覧・使用例を記載.
- Data Augmentation のデバッグ可視化機能を追加. `augmentation.debug_save` で 1 エポック目の最初の N 枚を bbox 付きで保存. ([#404](https://github.com/kurorosu/pochidetection/pull/404))
  - `build_data_loaders()` の augmentation 設定を `hasattr` から `isinstance(BaseCocoDataset)` チェックに改善.
- `process_frames()`, `draw_cv2()`, `_build_phase_summary()`, `_draw_overlay_text()` のテストを追加 (19 テスト). ([#405](https://github.com/kurorosu/pochidetection/pull/405))

### Changed
- 無し

### Fixed
- 無し

### Removed
- 無し

## Archived Changelogs

- [0.13.x](changelogs/0.13.x.md)
- [0.12.x](changelogs/0.12.x.md)
- [0.11.x](changelogs/0.11.x.md)
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
