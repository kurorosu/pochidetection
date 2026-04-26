# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し

### Changed
- `DetectionConfig` を architecture 別 discriminated union (`RTDetrConfig` / `SSDLiteConfig` / `SSD300Config`) に分離. SSD 系で `model_name` 等を指定すると警告ではなく `ValidationError` で弾く. ((NA.))

### Fixed
- 無し

### Removed
- 無し

## [0.19.0] - 2026-04-26

### Added
- `pochi prepare-demo` サブコマンドを追加. RT-DETR COCO プリトレイン (`PekingU/rtdetr_r50vd`) のダウンロード → ONNX エクスポート → TensorRT FP16 engine ビルドを 1 コマンドで実行し, `pochi serve -m work_dirs/<run>/best/model_fp16.engine` で WebAPI を即起動できる成果物を生成. TensorRT 必須 (未インストール時は副作用ゼロで早期 exit). ([#594](https://github.com/kurorosu/pochidetection/pull/594))

### Changed
- `scripts/rtdetr/train.py` の `processor_holder = []` リスト共有パターンを解消. model 構築を `_setup_training` 直下に移し, processor を `dataset_factory` のクロージャで参照する形に変更. SSD 系と同じ `partial(...)` 流儀に統一. ([#608](https://github.com/kurorosu/pochidetection/pull/608))
- 3 アーキ (RT-DETR / SSDLite / SSD300) の `_create_pytorch_backend` から `device → eval → FP16 → backend ラップ` の共通定型を `pochidetection/inference/builder.py::build_pytorch_backend` に集約. 各スクリプトはモデル構築のみを担当. ([#609](https://github.com/kurorosu/pochidetection/pull/609))
- SSD300 と SSDLite の `train.py` を `pochidetection/training/ssd.py::train_ssd` に集約. 各スクリプトはモデルクラスとアーキ名のラベルを渡すだけの 22 行に縮小 (元 84 行). ([#610](https://github.com/kurorosu/pochidetection/pull/610))
- F1ConfidencePlotter / PRCurvePlotter で 4 箇所に散らばっていた precision/scores の `-1 → NaN` 置換を `pochidetection/visualization/plotters/precision_utils.py::replace_invalid_with_nan` に集約. F1 plotter 側の冗長な mask 再計算も解消. ([#612](https://github.com/kurorosu/pochidetection/pull/612))
- `api/backends.py::create_detection_backend` のシグネチャから dead pass-through だった `config_path` 引数を削除し, config 解決と pretrained 経路の特例処理を `api/app.py::build_engine` 側に集約. WebAPI 経路で `resolve_and_build_pipeline` には常に `config_path=None` を渡す形で意味論を明示. ([#613](https://github.com/kurorosu/pochidetection/pull/613))
- `tensorrt/export.py::export_trt()` ラッパーを削除し, 呼び出し側 (`cli/commands/export.py` / `prepare_demo.py`) で直接 `TensorRTExporter().export()` を呼ぶ形に変更. ラッパーが追加していたログは exporter 側と完全重複していたため移植不要. デフォルト出力パス生成と TRT エクスポート失敗時の `sys.exit` を CLI レイヤに整理. ([#614](https://github.com/kurorosu/pochidetection/pull/614))
- `LoggerManager` の `_initialized` フラグ防御を廃止. インスタンス属性の初期化を `__init__` から `__new__` の Singleton 生成枝に集約し, `__init__` 自体を削除. クラスレベルの type annotation で mypy 整合も維持. ([#616](https://github.com/kurorosu/pochidetection/pull/616))
- `LoggerManager._create_handler` の `formatter` ローカル変数に `logging.Formatter` の型注釈を追加し, `colorlog` 有無の分岐で `ColoredFormatter` / `Formatter` のいずれを代入しても mypy 不整合エラーが出ないよう整理. 動作変更なし. ([#617](https://github.com/kurorosu/pochidetection/pull/617))
- `scripts/ssd300/infer.py` の `_unsupported_trt` / `_unsupported_onnx` の戻り値型を `SsdPyTorchBackend` から `NoReturn` に変更. 「常に raise する」意図を型として明示し, 呼び出し側 (`BackendFactories`) は `NoReturn` の bottom 性質により無修正で互換. ([#624](https://github.com/kurorosu/pochidetection/pull/624))
- `pyproject.toml` の mypy overrides に `onnxruntime` / `plotly` / `pynvml` / `tensorrt` を追加し, stub 不在系の `import-untyped` を一括 ignore. ([#625](https://github.com/kurorosu/pochidetection/pull/625))
- `core/letterbox.py::apply_letterbox` を `@overload` 化し PIL/Tensor 入出力の型推論を呼び出し側で正しく解決. ([#626](https://github.com/kurorosu/pochidetection/pull/626))
- `training/loop.py` の `DatasetFactory` を `Callable[..., Dataset[DatasetSampleDict]]` に正確化し, `partial(XxxDataset, ...)` の covariance 不整合を解消. ([#627](https://github.com/kurorosu/pochidetection/pull/627))
- `RTDetrPipeline.processor` の型を `RTDetrImageProcessor` から `RTDetrPostProcessor` Protocol に切り替え, テスト用 dummy processor の structural subtype 不整合を解消. ([#628](https://github.com/kurorosu/pochidetection/pull/628))
- 個別 mypy エラー 30 件を一掃 (Optional ガード / `cast` / `cv2.VideoWriter_fourcc` の `# type: ignore[attr-defined]` 等). `uv run mypy .` 0 件達成. ([#629](https://github.com/kurorosu/pochidetection/pull/629))
- pre-commit の mypy hook を `mirrors-mypy` から `local` + `uv run mypy` に切替. isolated venv の依存欠落問題を解消し commit 時にも全件型チェックが効くように. `core/letterbox.py` の `# type: ignore[overload-cannot-match]` 暫定 workaround も削除. ([#630](https://github.com/kurorosu/pochidetection/pull/630))
- `core/types.py::BuildPipelineFn` の戻り値型を `Any` から `PipelineContext` に具体化し, `build_pipeline()` 戻り値の型推論を有効化. ([#631](https://github.com/kurorosu/pochidetection/pull/631))
- `api/backends.py` に `BackendName = Literal["pytorch", "onnx", "tensorrt"]` Type alias を導入し, `detect_backend_from_model` 戻り値型と `_BACKEND_CLASSES` キー型を一元化. `get_args(BackendName)` でキー網羅を起動時 assert 検証. ([#632](https://github.com/kurorosu/pochidetection/pull/632))

### Fixed
- 無し

### Removed
- `IDetectionDataset` から未使用の抽象メソッド `get_categories()` / `get_num_classes()` / `get_category_names()` を削除. `BaseCocoDataset` の対応する実装と `_categories` 属性, 関連テストも合わせて削除. ([#611](https://github.com/kurorosu/pochidetection/pull/611))

## Archived Changelogs

- [0.18.x](changelogs/0.18.x.md)
- [0.17.x](changelogs/0.17.x.md)
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
