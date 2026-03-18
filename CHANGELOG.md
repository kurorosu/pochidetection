# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## Unreleased

### Added
- 無し

### Changed
- 無し

### Removed
- 無し

### Fixed
- 動画/Webcam/RTSP 推論で `-m` 未指定時に `work_dirs` のモデルを使おうとしてクラッシュするバグを修正. プリトレインモデルへ正しくフォールバックするように変更. (N/A.)

## v0.12.0 (2026-03-18)

### Added
- SSD300 VGG16 の学習実装を追加. `architecture = "SSD300"` で SSD300 学習が可能に. ([#352](https://github.com/kurorosu/pochidetection/pull/352))
- SSD300 PyTorch 推論バックエンドを追加. `architecture = "SSD300"` で SSD300 推論が可能に. ([#353](https://github.com/kurorosu/pochidetection/pull/353))
- 動画ファイル推論を追加. `pochi infer -d video.mp4` で .mp4/.avi/.mov の動画を入力可能に. `--interval N` で N フレーム間隔の推論にも対応. ([#356](https://github.com/kurorosu/pochidetection/pull/356))
  - モデル未指定時は RT-DETR COCO プリトレインモデルで推論. `local_files_only` 設定によるオフライン対応付き.
- リアルタイム推論を追加. `pochi infer -d 0` で Webcam, `-d rtsp://...` で RTSP ストリームに対応. `--record output.mp4` で表示と同時に録画可能. ([#357](https://github.com/kurorosu/pochidetection/pull/357))
  - FPS オーバーレイ表示, `q` キーで終了, `Ctrl+C` でも安全に停止.

### Changed
- SSDLite/SSD300 共通の `SsdPyTorchBackend` と `SsdPipeline` を抽出し, 推論バックエンド・パイプラインを共通化. ([#353](https://github.com/kurorosu/pochidetection/pull/353))
- SSDLite/SSD300 の検証ロジック・BN ヘルパー (`_validate`, `_save_bn_states`, `_restore_bn_states`) を `scripts/ssd/validation.py` に共通化. ([#355](https://github.com/kurorosu/pochidetection/pull/355))

### Removed
- `SSDLitePyTorchBackend` を `SsdPyTorchBackend` に統合し削除. ([#353](https://github.com/kurorosu/pochidetection/pull/353))
- `SSDLitePipeline` を `SsdPipeline` に統合し削除. ([#353](https://github.com/kurorosu/pochidetection/pull/353))

### Fixed
- 無し

## Archived Changelogs

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
