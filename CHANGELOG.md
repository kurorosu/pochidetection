# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## Unreleased

### Added
- 無し

### Changed
- `ModelOutputDict` をアーキテクチャ別のサブ型 (`TransformerModelOutputDict`, `SSDModelOutputDict`) に分離し, 各モデルの `forward()` 戻り値を型安全に. ([#379](https://github.com/kurorosu/pochidetection/pull/379))
- `IDetectionPipeline` に Generic 型パラメータ (`TPreprocessed`, `TInferred`) を導入し, パイプライン実装のフェーズ間データ型を明示. ([#380](https://github.com/kurorosu/pochidetection/pull/380))
- `IInferenceBackend` から `synchronize()` を削除し, デバイス同期を各バックエンドの `infer()` に統合. ISP 違反を解消. ([#381](https://github.com/kurorosu/pochidetection/pull/381))
- CLI の `_resolve_train` / `_resolve_infer` の if/elif 分岐をレジストリパターン (`cli/registry.py`) に変更. 新アーキテクチャ追加時に既存コードの修正が不要に. ([#382](https://github.com/kurorosu/pochidetection/pull/382))
- `run_training_loop()` を `TrainingLoop` クラスに責務分離. `EpochResult` dataclass でエポック結果を構造化し, ログ出力・履歴記録・TensorBoard・スケジューラ更新・Early Stopping を個別メソッドに分割. ([#383](https://github.com/kurorosu/pochidetection/pull/383))

### Removed
- `IInferenceBackend.synchronize()` 抽象メソッドを削除. CPU バックエンドに不要な空実装を強制していた ISP 違反を解消. ([#381](https://github.com/kurorosu/pochidetection/pull/381))

### Fixed
- 無し

## v0.12.1 (2026-03-18)

### Added
- 無し

### Changed
- `IFrameSource`, `IFrameSink` にコンテキストマネージャプロトコル (`__enter__`/`__exit__`) を追加. `with` 文でリソースを安全に解放可能に. ([#377](https://github.com/kurorosu/pochidetection/pull/377))
- 動画/ストリーム推論 (`_run_video_infer`, `_run_stream_infer`) のリソース管理を手動 `release()` から `with` 文に変更. ([#377](https://github.com/kurorosu/pochidetection/pull/377))

### Removed
- 無し

### Fixed
- 動画/Webcam/RTSP 推論で `-m` 未指定時に `work_dirs` のモデルを使おうとしてクラッシュするバグを修正. プリトレインモデルへ正しくフォールバックするように変更. ([#360](https://github.com/kurorosu/pochidetection/pull/360))
- Webcam/RTSP 推論で `q` キーまたは `Ctrl+C` で中断した際に完了サマリー (処理フレーム数, 経過時間, 平均 FPS) が出力されない問題を修正. ([#375](https://github.com/kurorosu/pochidetection/pull/375))
- Webcam/RTSP 推論中に 100 フレームごとの進捗ログが不要に出力される問題を修正. 動画ファイル推論のみに限定. ([#376](https://github.com/kurorosu/pochidetection/pull/376))
- `BaseCocoDataset.__getitem__()` で COCO JSON の画像情報に `id` または `file_name` フィールドが欠落している場合, 素の `KeyError` ではなくインデックスと画像情報を含む明確なエラーメッセージを出力するように改善. ([#377](https://github.com/kurorosu/pochidetection/pull/377))

## Archived Changelogs

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
