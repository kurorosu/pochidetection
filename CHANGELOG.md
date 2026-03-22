# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- `--record` オプションで録画中のストリーム推論時に, FPS オーバーレイの最下段に赤文字で `REC` を表示. ([#423](https://github.com/kurorosu/pochidetection/pull/423))
- リアルタイム推論の FPS オーバーレイに GPU 使用率・VRAM 使用量・CPU 使用率を表示. 30 フレームごとに更新. ([#424](https://github.com/kurorosu/pochidetection/pull/424))
  - `psutil`, `nvidia-ml-py` を依存に追加. `ResourceUsage` dataclass と `get_resource_usage()` を `utils/resource_monitor.py` に実装.
- リアルタイム推論中に `o` キーでオーバーレイの表示/非表示をトグルする機能を追加. ([#427](https://github.com/kurorosu/pochidetection/pull/427))
- リアルタイム推論中に `h` キーで画面右下にキーバインドヘルプ (`q:Quit s:Settings o:Status h:Help`) を表示/非表示する機能を追加. 初期状態はヘルプ表示・ステータス非表示. ([#429](https://github.com/kurorosu/pochidetection/pull/429))
- `pytest.mark.slow` マーカーを導入し, 時間のかかるテスト (CLI subprocess, TensorRT エクスポート, ONNX エクスポート, モデル新規初期化) を分類. 通常テスト実行時間を 84s → 15s に短縮 (82%削減). ([#432](https://github.com/kurorosu/pochidetection/pull/432))
  - SSD300Model / SSDLiteModel の session スコープフィクスチャを追加し, モデル初期化コストを共有化.
  - `tests/docs/slow_tests.md` に slow テスト一覧と理由を記載.

### Changed
- 無し

### Fixed
- ストリーム録画の再生時間が実際の録画時間と一致しない問題を修正. `LazyVideoWriter` を導入し, 最初の 10 フレームで実測 fps を推定して VideoWriter を初期化する方式に変更. N/A.

### Removed
- 無し

## v0.14.1 (2026-03-22)

### Added
- 無し

### Changed
- `build_data_loaders()` から augmentation 設定ロジックを分離し, `_apply_augmentation_to_dataset()` トップレベル関数に切り出し. `build_data_loaders()` のシグネチャを簡素化. ([#410](https://github.com/kurorosu/pochidetection/pull/410))
- `_run_stream_infer()` と `_run_video_infer()` のモデル解決・パイプライン構築ロジックを `_resolve_and_setup_pipeline()` に共通化. ([#411](https://github.com/kurorosu/pochidetection/pull/411))
- `_ResolvedPipeline.ctx` の型注釈を `Any` から `PipelineContext` に変更. `cli/registry.py` の `PipelineContext` インポートを `TYPE_CHECKING` ブロックに移動し循環インポートを解消. ([#416](https://github.com/kurorosu/pochidetection/pull/416))
- 画像推論 (`infer()`) のプリトレイン判定ロジックを `_resolve_and_setup_pipeline()` に統合. ([#419](https://github.com/kurorosu/pochidetection/pull/419))
- `SetupPipelineFn` の二重定義を `scripts/common/types.py` に集約. `_resolve_and_setup_pipeline()` を `cli/commands/infer.py` → `scripts/common/inference.py` に移動し, 下位→上位層の遅延インポートを解消. ([#420](https://github.com/kurorosu/pochidetection/pull/420))
- `resolve_and_setup_pipeline()` 内の遅延インポート (`resolve_setup_pipeline`, `PRETRAINED_CONFIG_PATH`, `ConfigLoader`) をトップレベルに移動. 循環回避のための遅延インポートがゼロに. ([#421](https://github.com/kurorosu/pochidetection/pull/421))
  - `inference.py` と `work_dir.py` の不要な遅延インポートを削除.
  - `augmentation.py` の `ImageDraw` をトップレベルに移動.
  - 正当な遅延インポートに why コメントを追加.

### Fixed
- `StreamReader.apply_camera_settings()` で `logger=None` 時にカメラ設定 (`cap.set()`) が適用されないバグを修正. logger チェックをログ出力部分のみに分離. ([#408](https://github.com/kurorosu/pochidetection/pull/408))
- `InferenceSaver.save_crops()` で bbox が画像外にはみ出す場合のクリッピングを追加. 面積ゼロの bbox はスキップ. ([#409](https://github.com/kurorosu/pochidetection/pull/409))

### Removed
- 無し

## Archived Changelogs

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
