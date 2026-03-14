# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- CLI 統合テストを追加. 引数パース (train/infer/export), アーキテクチャ別ディスパッチ (`_resolve_train`/`_resolve_infer`), コマンド間パス引き継ぎ (`resolve_config_path`), `run_infer` バリデーションの計 22 テストケース. ([#324](https://github.com/kurorosu/pochidetection/pull/324).)
- `write_reports()` の統合テストを追加. 正常検出, 空検出, GT アノテーション不在, config 保存の計 7 テストケース. ([#325](https://github.com/kurorosu/pochidetection/pull/325).)
- `INT8Calibrator` のキャリブレーションフローテストを追加. 複数バッチシーケンス, バッチサイズ未満の画像枚数, キャッシュの別インスタンス再利用の計 3 テストケース. (N/A.)

### Changed
- 無し.

### Removed
- 無し.

### Fixed
- 無し.

## v0.10.1 (2026-03-14)

### Added
- 無し.

### Changed
- `IInferenceBackend` を `Generic[TOutput]` に変更し, 全バックエンド実装の `infer()` 入出力型を具体化. `Any` を排除し mypy による型不整合検出を可能にした. ([#313](https://github.com/kurorosu/pochidetection/pull/313).)
- 学習スクリプトの `logger: Any` 型アノテーションを `logging.Logger` に修正. `Validator` Protocol を含む全 9 箇所を修正. mypy の `work_dirs/` 除外設定を追加. ([#314](https://github.com/kurorosu/pochidetection/pull/314).)
- `RTDetrPipeline` の `processor: Any` を `RTDetrImageProcessor` に修正. `processor_holder: list[Any]` も同様に具体化. ([#315](https://github.com/kurorosu/pochidetection/pull/315).)
- 設定辞書 `config: dict[str, Any]` を `DetectionConfigDict` (TypedDict) に置換. `ImageSizeDict` も導入し, mypy によるキー名・値型の静的チェックを可能にした. 全 17 ソースファイル + 6 テストファイルを修正. ([#316](https://github.com/kurorosu/pochidetection/pull/316).)
- エクスポートスクリプトの `except Exception` を `(OSError, ValueError, RuntimeError)` 等の具体的な例外型に絞り込み. 想定外の例外を握りつぶさないように改善. ([#317](https://github.com/kurorosu/pochidetection/pull/317).)
- `type: ignore` コメント 6 箇所を解消. `DataLoader` 型引数の明示化, `SSD` 型アノテーション追加, `DetectionConfigDict` 導入による `[index]` 抑制解消. 条件付き基底クラス (`calibrator.py`) の 1 箇所は mypy の制約により残存. ([#318](https://github.com/kurorosu/pochidetection/pull/318).)
- `IDetectionModel.forward()` の戻り値 `dict[str, Any]` を `ModelOutputDict` (TypedDict) に変更. `RTDetrModel`, `SSDLiteModel` も同様に適用. ([#319](https://github.com/kurorosu/pochidetection/pull/319).)
- `IDetectionDataset.__getitem__()` の戻り値 `dict[str, Any]` を `DatasetSampleDict` (TypedDict) に変更. `BaseCocoDataset`, `CocoDetectionDataset`, `SsdCocoDataset` も同様に適用. ([#320](https://github.com/kurorosu/pochidetection/pull/320).)
- `allocate_bindings()` の `engine: Any` / `context: Any` を `trt.ICudaEngine` / `trt.IExecutionContext` に変更. `TYPE_CHECKING` ブロックでインポートし mypy による型チェックを有効化. ([#322](https://github.com/kurorosu/pochidetection/pull/322).)

### Removed
- 無し.

### Fixed
- `EarlyStopping._is_improvement()` の `assert` を明示的な `RuntimeError` に置き換え. `python -O` 実行時にもガードが有効になるよう改善. ([#321](https://github.com/kurorosu/pochidetection/pull/321).)

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
