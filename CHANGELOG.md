# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 推論時 preprocess 後画像のデバッグ保存を追加. top-level `infer_debug_save_count` (既定 1) で制御し, CLI / 動画 / カメラ / WebAPI の全経路で先頭 N 枚を `{output_dir}/infer_debug/` へ保存. ([#577](https://github.com/kurorosu/pochidetection/pull/577))

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
- Pipeline route の命名整理: `pipelines/` の動詞を役割別 (`build_*` / `create_*` / `configure_*` / `resolve_*`) に統一し, `setup_pipeline` → `build_pipeline_from_spec`, `resolve_and_setup_pipeline` → `resolve_and_build_pipeline`, `setup_cudnn_benchmark` → `configure_cudnn_benchmark` にリネーム. ((NA.))
  - `ResolvedPipeline.ctx` を `.context` に改名し, orchestration の public parameter の `ctx: InferenceContext` も `context: InferenceContext` に統一. 短い local スコープの `ctx` は維持.

### Fixed
- 無し

### Removed
- 無し

## [0.17.0] - 2026-04-19

### Added
- 学習時 preprocess に letterbox (アスペクト比維持 + padding) リサイズを追加. top-level `letterbox: bool = True` フラグで制御し, SSDLite / SSD300 / RT-DETR 全アーキで train/infer 分布を一致させる. `core/letterbox.py` を PIL/Tensor 多態 API で新規作成し, #445 推論側の再利用を前提にした 2 層設計 (core + v2 Transform) を採用. ([#566](https://github.com/kurorosu/pochidetection/pull/566))
- 推論側 pipeline preprocess / postprocess に letterbox を組み込み, `core/letterbox.py` を CPU / GPU 両経路 (`gpu_preprocess_tensor`) から再利用. bbox は letterbox 逆変換 (`(box - pad) / scale`) で元画像座標に戻すため, レスポンスの bbox スキーマは変更なし. 極端なアスペクト比画像 (例: 1920x480) でも正しい座標が返る. `config["letterbox"]=False` で従来挙動に戻せる. ([#567](https://github.com/kurorosu/pochidetection/pull/567))

### Changed
- `IDetectionPipeline.run()` に `threshold` 引数を追加し, WebAPI `POST /api/v1/detect` のリクエスト毎 `score_threshold` がそのまま下限として効くように変更. backend 側の 2 段フィルタを撤廃. ([#564](https://github.com/kurorosu/pochidetection/pull/564))
- 学習画像のデバッグ保存を top-level `debug_save_count` (既定値 `10`) に昇格. augmentation の有無に関わらず発火し, 保存先は `{work_dir}/{run}/train_debug/train_XXXX.jpg`. letterbox / preprocess の silent bug 目視検知用. ([#565](https://github.com/kurorosu/pochidetection/pull/565))

### Fixed
- 無し

### Removed
- 無し

## Archived Changelogs

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
