# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し.

### Changed
- CLI の description を汎用化し, ヘルプに SSDLite の使用例と export の RT-DETR 限定注記を追加. `export` / `export-trt` コマンドに SSDLite 設定ファイル指定時のガードを追加. ([#209](https://github.com/kurorosu/pochidetection/pull/209).)
- `DetectionConfig.architecture` に `field_validator(mode="before")` を追加し, 大文字小文字を問わず `"RTDetr"` / `"SSDLite"` に正規化. ([#210](https://github.com/kurorosu/pochidetection/pull/210).)
- `WorkspaceManager.save_config()` が元のファイル名を保持してコピーするように変更. `resolve_config_path()` が `config.py` 以外の `.py` ファイルも自動検出するように拡張. ([#211](https://github.com/kurorosu/pochidetection/pull/211).)
- `TensorRTExporter.export()` のメモリプール制限を `build_memory` パラメータとして外部化. CLI に `--build-memory` オプションを追加. (N/A.)

### Fixed
- 無し.

### Removed
- 無し.

## v0.6.4 (2026-03-08)

### Added
- 無し.

### Changed
- `SSDLiteModel` と `ssdlite/infer.py` の docstring に, NMS が torchvision 内部で自動適用される旨の Note を追加. ([#205](https://github.com/kurorosu/pochidetection/pull/205).)
- SSDLite 推論を `SSDLitePipeline` クラスに抽出し, RT-DETR の `DetectionPipeline` と同じ 3 フェーズ構成に統一. 共通インターフェース `IDetectionPipeline` を新設し, 両パイプラインが実装. ([#206](https://github.com/kurorosu/pochidetection/pull/206).)
- `SsdCocoDataset` の画像リサイズとボックス座標変換を `torchvision.transforms.v2` + `tv_tensors.BoundingBoxes` に移行し, 手動スケール計算を削除. 推論側 (`ssdlite/infer.py`, `SSDLitePipeline`) の transform も v2 に統一. ([#207](https://github.com/kurorosu/pochidetection/pull/207).)

### Fixed
- 無し.

### Removed
- 無し.

## Archived Changelogs

- [0.6.x](changelogs/0.6.x.md)
- [0.5.x](changelogs/0.5.x.md)
- [0.4.x](changelogs/0.4.x.md)
- [0.3.x](changelogs/0.3.x.md)
- [0.2.x](changelogs/0.2.x.md)
- [0.1.x](changelogs/0.1.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
