# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し.

### Changed
- 無し.

### Removed
- 無し.

### Fixed
- `Image.open()` のファイルハンドル未クローズを修正. `with` 文で明示的にクローズするよう変更 (`inference.py`, `base_coco_dataset.py`, `calibrator.py`). ([#329](https://github.com/kurorosu/pochidetection/pull/329).)
- ONNX 検証セッション (`InferenceSession`) の未クローズを修正. `try/finally` で明示的に解放するよう変更 (`validation.py`). (N/A.)

## v0.10.2 (2026-03-14)

### Added
- CLI 統合テストを追加. 引数パース (train/infer/export), アーキテクチャ別ディスパッチ (`_resolve_train`/`_resolve_infer`), コマンド間パス引き継ぎ (`resolve_config_path`), `run_infer` バリデーションの計 22 テストケース. ([#324](https://github.com/kurorosu/pochidetection/pull/324).)
- `write_reports()` の統合テストを追加. 正常検出, 空検出, GT アノテーション不在, config 保存の計 7 テストケース. ([#325](https://github.com/kurorosu/pochidetection/pull/325).)
- `INT8Calibrator` のキャリブレーションフローテストを追加. 複数バッチシーケンス, バッチサイズ未満の画像枚数, キャッシュの別インスタンス再利用の計 3 テストケース. ([#326](https://github.com/kurorosu/pochidetection/pull/326).)

### Changed
- `rtdetr_model` fixture を eval モードで初期化するよう変更. テスト間の状態干渉を防止. ([#327](https://github.com/kurorosu/pochidetection/pull/327).)

### Removed
- 無し.

### Fixed
- TensorRT INT8 キャリブレーション関連の DeprecationWarning を pytest filterwarnings で抑制. ([#327](https://github.com/kurorosu/pochidetection/pull/327).)

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
