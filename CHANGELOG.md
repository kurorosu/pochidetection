# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- `pochi prepare-demo` サブコマンドを追加. RT-DETR COCO プリトレイン (`PekingU/rtdetr_r50vd`) のダウンロード → ONNX エクスポート → TensorRT FP16 engine ビルドを 1 コマンドで実行し, `pochi serve -m work_dirs/<run>/best/model_fp16.engine` で WebAPI を即起動できる成果物を生成. TensorRT 必須 (未インストール時は副作用ゼロで早期 exit). ([#594](https://github.com/kurorosu/pochidetection/pull/594))

### Changed
- `scripts/rtdetr/train.py` の `processor_holder = []` リスト共有パターンを解消. model 構築を `_setup_training` 直下に移し, processor を `dataset_factory` のクロージャで参照する形に変更. SSD 系と同じ `partial(...)` 流儀に統一. ([#608](https://github.com/kurorosu/pochidetection/pull/608))
- 3 アーキ (RT-DETR / SSDLite / SSD300) の `_create_pytorch_backend` から `device → eval → FP16 → backend ラップ` の共通定型を `pochidetection/inference/builder.py::build_pytorch_backend` に集約. 各スクリプトはモデル構築のみを担当. ([#609](https://github.com/kurorosu/pochidetection/pull/609))
- SSD300 と SSDLite の `train.py` を `pochidetection/training/ssd.py::train_ssd` に集約. 各スクリプトはモデルクラスとアーキ名のラベルを渡すだけの 22 行に縮小 (元 84 行). ([#610](https://github.com/kurorosu/pochidetection/pull/610))
- F1ConfidencePlotter / PRCurvePlotter で 4 箇所に散らばっていた precision/scores の `-1 → NaN` 置換を `pochidetection/visualization/plotters/precision_utils.py::replace_invalid_with_nan` に集約. F1 plotter 側の冗長な mask 再計算も解消. ([#612](https://github.com/kurorosu/pochidetection/pull/612))
- `api/backends.py::create_detection_backend` のシグネチャから dead pass-through だった `config_path` 引数を削除し, config 解決と pretrained 経路の特例処理を `api/app.py::build_engine` 側に集約. WebAPI 経路で `resolve_and_build_pipeline` には常に `config_path=None` を渡す形で意味論を明示. ([#613](https://github.com/kurorosu/pochidetection/pull/613))
- `tensorrt/export.py::export_trt()` ラッパーを削除し, 呼び出し側 (`cli/commands/export.py` / `prepare_demo.py`) で直接 `TensorRTExporter().export()` を呼ぶ形に変更. ラッパーが追加していたログは exporter 側と完全重複していたため移植不要. デフォルト出力パス生成と TRT エクスポート失敗時の `sys.exit` を CLI レイヤに整理. ((NA.))

### Fixed
- 無し

### Removed
- `IDetectionDataset` から未使用の抽象メソッド `get_categories()` / `get_num_classes()` / `get_category_names()` を削除. `BaseCocoDataset` の対応する実装と `_categories` 属性, 関連テストも合わせて削除. ([#611](https://github.com/kurorosu/pochidetection/pull/611))

## [0.18.0] - 2026-04-25

### Added
- 推論時 preprocess 後画像のデバッグ保存を追加. top-level `infer_debug_save_count` (既定 1) で制御し, CLI / 動画 / カメラ / WebAPI の全経路で先頭 N 枚を `{output_dir}/infer_debug/` へ保存. ([#577](https://github.com/kurorosu/pochidetection/pull/577))
- `DetectResponse` に GPU メトリクス 3 フィールド (`gpu_clock_mhz` / `gpu_vram_used_mb` / `gpu_temperature_c`) を追加. `api/gpu_clock.py` を `api/gpu_metrics.py` にリネームし, VRAM 使用量 / 温度取得関数を新設して handle をキャッシュ共有. ([#586](https://github.com/kurorosu/pochidetection/pull/586))
- `POST /api/v1/detect` の `phase_times_ms` に API boundary 計測キー `api_preprocess_ms` (deserialize + cvtColor) / `api_postprocess_ms` (results 組み立て + DetectResponse 構築) を追加. 全フェーズ合計が `e2e_time_ms` と概ね一致する. INFO ログにも `api_pre` / `api_post` を併記. ([#589](https://github.com/kurorosu/pochidetection/pull/589))
- `pochi serve` の `-m` を省略可能化し, モデル未指定時に RT-DETR COCO プリトレインモデルで起動できるように (`pochi infer` と同等の体験). 学習済みモデル無しの環境で API 動作確認 / 他ツール連携デモが即座に行える. ([#590](https://github.com/kurorosu/pochidetection/pull/590))
- `POST /api/v1/detect` の bbox が letterbox 逆変換を経た元画像座標系で返ることを `@pytest.mark.slow` で検証. 1280x720 入力で max(x2/y2) が target_hw を超えることを assert し, 逆変換欠落の回帰を検知する. ([#592](https://github.com/kurorosu/pochidetection/pull/592))

### Changed
- Pipeline の letterbox 幾何パラメータ (`_last_letterbox_params`) をインスタンス属性から preprocess 戻り値経由の request-scoped 受け渡しに変更し, 同一 pipeline を複数 thread から並行呼出しても bbox 逆変換が混線しないようにした. ([#576](https://github.com/kurorosu/pochidetection/pull/576))
- `scripts/{rtdetr,ssdlite,ssd300}/infer.py::_setup_pipeline` の共通 boilerplate を `pipelines/builder.py::setup_pipeline` + `ArchitectureSpec` dataclass に集約し, 各スクリプトの本体行数を 50-70% 削減. ([#578](https://github.com/kurorosu/pochidetection/pull/578))
- `pipelines/builder.py` の pipeline 構築部を `model_path` / `runtime` / `backend_factory` / `context` / `spec` の 5 モジュールに分割し, 各モジュールで `__all__` 境界を明確化. builder.py は CLI batch フロー専用に縮小. ([#579](https://github.com/kurorosu/pochidetection/pull/579))
  - `tests/test_pipelines/test_builder.py` も責務単位に `test_model_path` / `test_runtime` / `test_spec` / `test_builder` の 4 ファイルへ分割.
- `pipelines/builder.py` を削除し CLI batch 推論フロー (画像ループ / レポート出力) を新設 `orchestration/` パッケージへ移設. `builder.infer` は `orchestration.run_batch_inference` にリネーム. ([#581](https://github.com/kurorosu/pochidetection/pull/581))
  - `tests/test_pipelines/test_builder.py` を `tests/test_orchestration/test_batch_inference.py` へ移設.
- 推論 route の命名整理: `_setup_pipeline` (→`build_pipeline`) / `_InferenceContext` / `_resolve_model_path` を public 化し, CLAUDE.md 命名規約との乖離を是正. ([#582](https://github.com/kurorosu/pochidetection/pull/582))
  - `cli/registry.py` の `resolve_infer` / `resolve_setup_pipeline` を `get_*_for_arch` にリネーム + `__all__` 宣言.
- 学習 route の命名整理: `cli/registry.py::resolve_train` を推論 route と揃えて `get_train_for_arch` にリネーム. `cli/commands/train.py` の `train_fn` ローカル変数から冗長な `_fn` を削除. ([#583](https://github.com/kurorosu/pochidetection/pull/583))
- Pipeline route の命名整理: `pipelines/` の動詞を役割別 (`build_*` / `create_*` / `configure_*` / `resolve_*`) に統一し, `setup_pipeline` → `build_pipeline_from_spec`, `resolve_and_setup_pipeline` → `resolve_and_build_pipeline`, `setup_cudnn_benchmark` → `configure_cudnn_benchmark` にリネーム. ([#585](https://github.com/kurorosu/pochidetection/pull/585))
  - `ResolvedPipeline.ctx` を `.context` に改名し, orchestration の public parameter の `ctx: InferenceContext` も `context: InferenceContext` に統一. 短い local スコープの `ctx` は維持.

### Fixed
- 無し

### Removed
- 無し

## Archived Changelogs

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
