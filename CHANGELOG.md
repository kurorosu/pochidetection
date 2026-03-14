# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

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

### Removed
- 無し.

### Fixed
- 無し.

## v0.10.0 (2026-03-14)

### Added
- `F1ConfidencePlotter` を追加. 学習時に信頼度閾値ごとの F1 スコア変化を `f1_confidence.html` として可視化. ([#251](https://github.com/kurorosu/pochidetection/pull/251).)
- `docs/evaluation_metrics.md` を追加. 物体検出の評価指標 (TP/FP/FN, Precision/Recall/F1, 混同行列, PR 曲線, F1-Confidence 曲線, NMS) の解説ドキュメント. ([#253](https://github.com/kurorosu/pochidetection/pull/253).)
- 推論時に使用した config ファイルを `inference_NNN/` ディレクトリにコピーする機能を追加. 推論条件の再現性を向上. ([#254](https://github.com/kurorosu/pochidetection/pull/254).)

### Changed
- `cli/main.py` を `parser.py` + `commands/` に分割. main.py を 50 行以下に縮小. ([#252](https://github.com/kurorosu/pochidetection/pull/252).)
- SSDLite ONNX/TensorRT バックエンドの後処理ロジック (`_generate_anchors`, `_postprocess`, `_decode_boxes`) を `postprocessing.py` に共通化. ~150行の重複を解消. ([#285](https://github.com/kurorosu/pochidetection/pull/285).)
- RT-DETR/SSDLite 推論エントリポイント `infer()` と `_run_inference()` を `scripts/common/inference.py` に共通化. 各アーキテクチャは `_setup_pipeline` のみ担当する設計に変更. ([#290](https://github.com/kurorosu/pochidetection/pull/290).)
- RT-DETR/SSDLite バックエンド生成ロジック (`_create_backend`, `_is_onnx_model`, `_is_tensorrt_model`) を `scripts/common/inference.py` に共通化. ファクトリコールバックパターンで分岐を単一箇所に集約. ([#302](https://github.com/kurorosu/pochidetection/pull/302).)
- RT-DETR/SSDLite パイプライン初期化ロジック (cudnn 設定, デバイス判定, LabelMapper/Visualizer/Saver 構築) を `scripts/common/inference.py` に共通化. `PipelineContext` NamedTuple, `setup_cudnn_benchmark()`, `resolve_device()`, `build_pipeline_context()` を追加. ([#303](https://github.com/kurorosu/pochidetection/pull/303).)
- RT-DETR/SSDLite 学習セットアップ (`_setup_training`) を `scripts/common/training.py` の `setup_training()` に共通化. ワークスペース作成, オプティマイザ, スケジューラ, mAP メトリクス初期化を単一箇所に集約. ([#304](https://github.com/kurorosu/pochidetection/pull/304).)
- RT-DETR/SSDLite PyTorch バックエンドの `synchronize()` メソッドを `inference/sync.py` の `synchronize_cuda()` に共通化. ([#305](https://github.com/kurorosu/pochidetection/pull/305).)
- RT-DETR/SSDLite ONNX バックエンドの `_resolve_providers()` を `inference/providers.py` の `resolve_providers()` に共通化. ([#306](https://github.com/kurorosu/pochidetection/pull/306).)
- RT-DETR/SSDLite ONNX エクスポーターの検証ロジック (`verify`) を `onnx/validation.py` の `verify_onnx_outputs()` に共通化. 構造検証, ONNX Runtime 推論, 出力比較, ログ出力を単一関数に集約. ([#307](https://github.com/kurorosu/pochidetection/pull/307).)
- RT-DETR/SSDLite 推論パイプラインの `run()` フェーズ計測ロジックを `IDetectionPipeline` に共通化. `_validate_phased_timer()`, `_measure()`, `phased_timer` プロパティを基底クラスに集約し, サブクラスの if/else 分岐を解消. ([#308](https://github.com/kurorosu/pochidetection/pull/308).)
- 全推論バックエンド (ONNX/TensorRT × RT-DETR/SSDLite) の入力検証パターンを `inference/validation.py` の `validate_inputs()` に共通化. ([#309](https://github.com/kurorosu/pochidetection/pull/309).)
- 全推論バックエンド (ONNX/TensorRT × RT-DETR/SSDLite) のモデルファイル検証パターン (存在確認, ファイル確認, 拡張子確認) を `inference/validation.py` の `validate_model_file()` に共通化. ([#310](https://github.com/kurorosu/pochidetection/pull/310).)
- 設定ファイルの共通パラメータ (クラス, データ, 学習, Early Stopping, デバイス, 閾値, ワークスペース) を `configs/_base.py` に抽出. `ConfigLoader` がベースと個別設定を自動マージし, `save_config` はマージ済み設定を展開して保存する構成に変更. ([#311](https://github.com/kurorosu/pochidetection/pull/311).)

### Removed
- 無し.

### Fixed
- 無し.

## Archived Changelogs

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
