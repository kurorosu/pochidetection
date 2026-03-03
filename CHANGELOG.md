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

## v0.4.2 (2026-03-04)

### Added
- なし.

### Changed
- `TensorRTBackend.infer()` の `wait_stream` をループ外に移動し, 冗長な同期オーバーヘッドを削減. ([#101](https://github.com/kurorosu/pochidetection/pull/101).)
- `infer.py` の `import torch` をインライン import からトップレベル import に移動. CLAUDE.md のコーディング規約に準拠. ([#102](https://github.com/kurorosu/pochidetection/pull/102).)

### Fixed
- なし.

### Removed
- `IDetectionModel` / `RTDetrModel` の `get_backbone_params()` / `get_head_params()` を削除. 未使用であり, differential learning rate は未実装のため. ([#100](https://github.com/kurorosu/pochidetection/pull/100).)

## Archived Changelogs

- [0.4.x](changelogs/0.4.x.md)
- [0.3.x](changelogs/0.3.x.md)
- [0.2.x](changelogs/0.2.x.md)
- [0.1.x](changelogs/0.1.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
