# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- HTTP body サイズ上限 middleware を追加し, 超過時に 413 を返す. デフォルト 64MB, `POCHI_MAX_BODY_SIZE` で上書き可能. ([#523](https://github.com/kurorosu/pochidetection/pull/523))

### Changed
- `api/state.py` を切り出し `routers/inference.py` の関数内遅延 import を解消. ([#520](https://github.com/kurorosu/pochidetection/pull/520))
- `api/constants.py` を新設し `MAX_PIXELS` / `_ALLOWED_DTYPES` を `schemas.py` から分離. ([#521](https://github.com/kurorosu/pochidetection/pull/521))
- CUDA Event を `__init__` キャッシュ化し, 推論毎の生成コストを撤廃. ([#522](https://github.com/kurorosu/pochidetection/pull/522))

### Fixed
- 無し

### Removed
- 無し

## [0.16.2] - 2026-04-19

### Added
- 無し

### Changed
- `pred_boxes` の `[0, 1]` 値域テストと, `score_threshold` / `confidence` / `MAX_PIXELS` の境界値テストを parametrize で追加. ([#509](https://github.com/kurorosu/pochidetection/pull/509))
- `PyTorchDetectionBackend` の実モデル E2E テストを `@pytest.mark.slow` で追加 (warmup / predict / set_class_names / get_model_info). MagicMock を使わない classical test. ([#510](https://github.com/kurorosu/pochidetection/pull/510))
- `IImageSerializer` Protocol を `RawArraySerializer` / `JpegSerializer` が明示継承するよう変更. ([#511](https://github.com/kurorosu/pochidetection/pull/511))
- `datasets/augmentation.py` の debug 画像色リストを `ColorPalette.DEFAULT_COLORS` 参照に統一. ([#512](https://github.com/kurorosu/pochidetection/pull/512))
- plotter 3 ファイルの `<!DOCTYPE html>` 横並びテンプレートを `plotters/constants.py` の `render_side_by_side_html()` に集約. ([#513](https://github.com/kurorosu/pochidetection/pull/513))
- SSD `infer()` を RTDetr と統一し dict 入力と `torch.no_grad()` を適用. ([#514](https://github.com/kurorosu/pochidetection/pull/514))
- `_safe_version()` とログ日付フォーマット / 色定義の重複を共通化. ([#515](https://github.com/kurorosu/pochidetection/pull/515))

### Fixed
- 無し

### Removed
- 無し

## Archived Changelogs

- [0.16.x](changelogs/0.16.x.md)
- [0.15.x](changelogs/0.15.x.md)
- [0.14.x](changelogs/0.14.x.md)
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
