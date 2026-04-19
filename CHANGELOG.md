# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- `tests/docs/testing_guide.md` 新設. テスト哲学 / Fixtures / Parametrize / slow marker / CUDA テスト等を網羅した開発者向けガイド. ([#552](https://github.com/kurorosu/pochidetection/pull/552))

### Changed
- `api/backends.py` の docstring を日本語化し Google style に統一. `create_detection_backend` の旧記述修正と `IDetectionBackend.close()` の no-op 許容方針明記を含む. ([#546](https://github.com/kurorosu/pochidetection/pull/546))
- `api/schemas.py` の `DetectRequest` docstring を 1 行に簡潔化し, 詳細を `Field(description=...)` に統合. ([#547](https://github.com/kurorosu/pochidetection/pull/547))
- `pochi infer / serve` の `-d` / `-m` help に具体例 (画像 / 動画 / カメラ ID / RTSP URL, `.onnx` / `.engine`) を追加. ([#548](https://github.com/kurorosu/pochidetection/pull/548))
- `docs/api_detect_inference_variance_investigation.md` に `## 更新履歴` セクションを追加. 冗長な「教訓と再発防止」セクションを削除. ([#549](https://github.com/kurorosu/pochidetection/pull/549))
- `pyproject.toml` の `slow` marker 説明を実モデルロード / GPU / TensorRT engine ビルド等の具体例付きに拡充. ([#550](https://github.com/kurorosu/pochidetection/pull/550))
- Issue テンプレートの Branch プレースホルダを統一 (`feature/` → `feat/`, `test_request.md` 末尾改行修正). ([#551](https://github.com/kurorosu/pochidetection/pull/551))
- `ConfigLoader._extract_config` docstring に `exec_module()` の挙動と security 前提 (trusted config のみ対象) を Notes / Warning として明記. ([#553](https://github.com/kurorosu/pochidetection/pull/553))
- `api/gpu_clock.py` の pynvml init / handle 取得失敗ログを INFO → WARNING に引き上げ. ([#554](https://github.com/kurorosu/pochidetection/pull/554))
- `IDetectionPipeline.pipeline_mode` property の `getattr` fallback を撤廃し, 基底 `__init__` で `_pipeline_mode` を明示初期化. ([#555](https://github.com/kurorosu/pochidetection/pull/555))
- `CocoDetectionDataset` の bbox 用 empty tensor に `dtype=torch.float32` を明示. ([#556](https://github.com/kurorosu/pochidetection/pull/556))
- `training/loop.py` の `len()` 呼び出しの `# type: ignore[arg-type]` を `cast(Sized, ...)` で解消. ([#557](https://github.com/kurorosu/pochidetection/pull/557))
- `BaseCocoDataset` の `_debug_save_*` private 属性に公開 property + setter を追加 (型 / 値検証付き). 学習ループ側も新 setter 経由に移行. ([#558](https://github.com/kurorosu/pochidetection/pull/558))
- `tests/test_onnx/conftest.py` で二重定義されていた `ssdlite_model` fixture を削除し, 親 `tests/conftest.py` に寄せて重複解消. ([#559](https://github.com/kurorosu/pochidetection/pull/559))
- `routers/inference.py` のログ整形ヘルパー (`format_phase` / `format_inference`) を `api/log_format.py` に切り出し. `_serializer_cache` を `threading.Lock` で保護し thread-safe 化. ([#560](https://github.com/kurorosu/pochidetection/pull/560))
- `tests/test_api/` の類似テストを `@pytest.mark.parametrize` で統合 (`create_serializer` / `NVMLError` 失敗経路 / `backend_name` 3 実装). ((NA.))

### Fixed
- 無し

### Removed
- 無し

## [0.16.3] - 2026-04-19

### Added
- HTTP body サイズ上限 middleware を追加し, 超過時に 413 を返す. デフォルト 64MB, `POCHI_MAX_BODY_SIZE` で上書き可能. ([#523](https://github.com/kurorosu/pochidetection/pull/523))

### Changed
- `api/state.py` を切り出し `routers/inference.py` の関数内遅延 import を解消. ([#520](https://github.com/kurorosu/pochidetection/pull/520))
- `api/constants.py` を新設し `MAX_PIXELS` / `_ALLOWED_DTYPES` を `schemas.py` から分離. ([#521](https://github.com/kurorosu/pochidetection/pull/521))
- CUDA Event を `__init__` キャッシュ化し, 推論毎の生成コストを撤廃. ([#522](https://github.com/kurorosu/pochidetection/pull/522))
- `configs/schemas.py` の `DetectionConfig` (Pydantic) と `DetectionConfigDict` (TypedDict) の使い分けを docstring に明文化. ([#524](https://github.com/kurorosu/pochidetection/pull/524))

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
