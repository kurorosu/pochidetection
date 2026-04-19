# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し

### Changed
- `pred_boxes` の `[0, 1]` 値域テストと, `score_threshold` / `confidence` / `MAX_PIXELS` の境界値テストを parametrize で追加. ([#509](https://github.com/kurorosu/pochidetection/pull/509))
- `PyTorchDetectionBackend` の実モデル E2E テストを `@pytest.mark.slow` で追加 (warmup / predict / set_class_names / get_model_info). MagicMock を使わない classical test. ([#510](https://github.com/kurorosu/pochidetection/pull/510))
- `IImageSerializer` Protocol を `RawArraySerializer` / `JpegSerializer` が明示継承するよう変更. (NA.)
- `datasets/augmentation.py` の debug 画像色リストを `ColorPalette.DEFAULT_COLORS` 参照に統一. (NA.)
- plotter 3 ファイルの `<!DOCTYPE html>` 横並びテンプレートを `plotters/constants.py` の `render_side_by_side_html()` に集約. (NA.)

### Fixed
- 無し

### Removed
- 無し

## [0.16.1] - 2026-04-19

### Added
- 無し

### Changed
- `api/gpu_clock.py` の `get_gpu_clock_mhz()` に対する単体テストを追加. pynvml 未インストール / 初期化失敗 / handle 取得失敗 / clock 取得失敗 / handle キャッシュ挙動の各分岐をカバー. ([#470](https://github.com/kurorosu/pochidetection/pull/470))
- `IDetectionBackend` の `get_model_info` / `set_class_names` / `warmup` / `close` / `backend_name` の動作テストを追加. 既存 phase_times テストも class 単位で整理. ([#473](https://github.com/kurorosu/pochidetection/pull/473))
- `IDetectionPipeline.pipeline_mode` プロパティの cpu / gpu 反映を RTDetr / SSD 両方で検証するテストを追加. ([#471](https://github.com/kurorosu/pochidetection/pull/471))
- `resolve_pipeline_mode()` の入力組合せを `@pytest.mark.parametrize` で網羅. ONNX + gpu 拒否時の `ValueError` メッセージも検証. ([#472](https://github.com/kurorosu/pochidetection/pull/472))
- `pochidetection/interfaces/pipeline.py` の `typing.Union` を Python 3.12+ の `X | Y` 記法に置換. `ImageInput` 型エイリアスを PEP 695 の `type` 文に書き換え. ([#475](https://github.com/kurorosu/pochidetection/pull/475))
- WebAPI テスト (`tests/test_api/`) のグローバル `_engine` 直接書き換えを `conftest.py` の autouse fixture で隔離. 各テストの try/finally 手動リセットを削除し, pytest-xdist 並列実行時の state 残留を防止. ([#476](https://github.com/kurorosu/pochidetection/pull/476))
- `IDetectionModel` に `num_classes` と `model` の抽象プロパティを追加. 3 実装 (RTDetr / SSD300 / SSDLite) は既にプロパティを持つため契約の明文化のみ. ([#477](https://github.com/kurorosu/pochidetection/pull/477))
- `IDetectionDataset` に `get_num_classes()` と `get_category_names()` の抽象メソッドを追加. `BaseCocoDataset` は既に実装済みのため契約の明文化のみ. ([#478](https://github.com/kurorosu/pochidetection/pull/478))
- `LoggerManager._create_logger` の `logger.handlers.clear()` を撤去し, 自前 handler に `_pochi_owned` マーカを付与して重複追加を回避. 外部 (pytest caplog 等) が追加した handler を破壊しない設計に修正. ([#479](https://github.com/kurorosu/pochidetection/pull/479))
- RTDetr / SSD の `_preprocess_gpu` 重複を `scripts/common/preprocess.py` の `gpu_preprocess_tensor` ヘルパーに共通化. 両 Pipeline はヘルパー呼び出しに置換し buffer 再利用の state のみ保持する設計に変更. ([#480](https://github.com/kurorosu/pochidetection/pull/480))
  - persisted buffer を常に float32 で維持. fp16 経路は戻り値 `pixel_values` のみ `.half()` キャストし, 次回 `copy_(uint8)` 時の dtype 事故を防止.
  - `tests/test_scripts/test_common_preprocess.py` にヘルパー単体テスト (shape 検証 / buffer 再利用 / fp16 / resize skip) を追加.
- `SSD300Model` / `SSDLiteModel` の `forward` / `save` / `load` 等の共通実装を `pochidetection/models/ssd_base.py` の `SSDModelBase` に集約. サブクラスは Template Method (`_create_torchvision_model`) で torchvision factory と weights のみ提供する設計に変更. ([#481](https://github.com/kurorosu/pochidetection/pull/481))
  - 両サブクラスの `__init__` を撤去. default 値も基底側に寄せる.
  - `SSDLiteModel` / `ssdlite/onnx_backend.py` / `ssdlite/tensorrt_backend.py` の `nms_iou_threshold` default を `0.55` → `0.5` に統一し, [#348](https://github.com/kurorosu/pochidetection/pull/348) 時点で残っていた統一漏れを解消. 通常経路 (`pochi train` / `pochi infer`) は config 経由で 0.5 が渡されるため実害なし.
  - Issue テンプレート (`.github/ISSUE_TEMPLATE/{feature,refactor,test,documentation}_request.md`) の Acceptance Criteria からチェックボックス (`- [ ]`) を撤去. 規約 (`.claude/rules/github-templates.md`) と整合させる.
- `RTDetrPipeline` / `SsdPipeline` を `pochidetection/pipelines/` 配下に移動. ([#485](https://github.com/kurorosu/pochidetection/pull/485))
- `scripts/common/` のライブラリ的モジュールを実態に合わせて再配置. ([#486](https://github.com/kurorosu/pochidetection/pull/486))
  - `preprocess.py` / `coco_classes.py` / `types.py` → `core/`
  - `saver.py` / `summary.py` / `detection_results_writer.py` / `visualizer.py` → `reporting/` (新設)
  - `inference.py` → `pipelines/builder.py`
  - `training.py` → `training/loop.py` (新設)
  - `video.py` → `utils/video.py`
  - `export_trt.py` → `tensorrt/export.py`
  - pipeline テスト / 関連テストを対応ディレクトリ (`tests/test_pipelines/` / `tests/test_core/` / `tests/test_reporting/` / `tests/test_training/` 等) に同期移動.
- `scripts/` を CLI エントリのみに整理. ([#487](https://github.com/kurorosu/pochidetection/pull/487))
  - `scripts/ssd/validation.py` → `training/validation.py` に移動 (ライブラリのため).
  - 空ディレクトリ (`scripts/common/` / `scripts/ssd/` / `scripts/rtdetr/inference/` / `scripts/ssd/inference/`) を撤去.
- `pipelines/builder.py` の public / private API を明確化. `__all__` を宣言し, 外部参照の無い関数 (`_run_inference` / `_write_reports` / `_resolve_model_path` / `_collect_image_files` / `_InferenceContext` 等) を `_` prefix に変更. public を上に private を下に配置する構成に並び替え. ([#489](https://github.com/kurorosu/pochidetection/pull/489))
- RTDetr / SSD の `_preprocess_gpu` と `forward` の docstring に tensor の shape / dtype / device / 値域を明記. ([#504](https://github.com/kurorosu/pochidetection/pull/504))
- `docs/api-server.md` に `/health`, `/version`, `/model-info`, `/backends` のレスポンススキーマ (JSON 例 + フィールド型表) を追記. モデル未ロード時の挙動 (`/health` は 200 で `model_loaded=false`, `/model-info` は 503) と `backend_versions` の動的構造も明記. ([#505](https://github.com/kurorosu/pochidetection/pull/505))
- RTDetr / SSD pipeline の `pipeline_mode` プロパティテストを `tests/test_pipelines/test_pipeline_mode.py` に parametrize で統合. 重複していた `test_pipeline_mode_property_returns_gpu` を撤去. `gpu_preprocess_tensor` の各単体テストに device パラメータ (`cpu` / `cuda`) を追加し, `pytest.mark.skipif(not torch.cuda.is_available(), ...)` で CUDA 環境での実機検証を可能化. ([#506](https://github.com/kurorosu/pochidetection/pull/506))
- `resolve_device()` の docstring を拡充し, backend 種別ごとの戻り値 (`actual_device` / `runtime_device`) の決定ロジックと ONNX で `runtime_device` を常に `"cpu"` に固定している理由 (ONNX Runtime が CPU numpy 入力を要求するため GPU preprocess に効果がない) を明記. `resolve_pipeline_mode()` が ONNX + `--pipeline gpu` を拒否する `ValueError` メッセージに具体的なフォールバック手順 (`--pipeline cpu` への切替, または PyTorch / TensorRT モデルへの切替) を追記. ([#507](https://github.com/kurorosu/pochidetection/pull/507))

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
