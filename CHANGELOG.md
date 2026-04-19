# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し

### Changed
- 無し

### Fixed
- 無し

### Removed
- 無し

## [0.16.3] - 2026-04-19

### Added
- HTTP body サイズ上限 middleware を追加し, 超過時に 413 を返す. デフォルト 64MB, `POCHI_MAX_BODY_SIZE` で上書き可能. ([#523](https://github.com/kurorosu/pochidetection/pull/523))

### Changed
- `api/state.py` を切り出し `routers/inference.py` の関数内遅延 import を解消. ([#520](https://github.com/kurorosu/pochidetection/pull/520))
- `api/constants.py` を新設し `MAX_PIXELS` / `_ALLOWED_DTYPES` を `schemas.py` から分離. ([#521](https://github.com/kurorosu/pochidetection/pull/521))
- CUDA Event を `__init__` キャッシュ化し, 推論毎の生成コストを撤廃. ([#522](https://github.com/kurorosu/pochidetection/pull/522))
- `configs/schemas.py` の `DetectionConfig` (Pydantic) と `DetectionConfigDict` (TypedDict) の使い分けを docstring に明文化. ([#524](https://github.com/kurorosu/pochidetection/pull/524))

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
