# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し

### Changed
- `api/gpu_clock.py` の `get_gpu_clock_mhz()` に対する単体テストを追加. pynvml 未インストール / 初期化失敗 / handle 取得失敗 / clock 取得失敗 / handle キャッシュ挙動の各分岐をカバー. (NA.)
- `IDetectionBackend` の `get_model_info` / `set_class_names` / `warmup` / `close` / `backend_name` の動作テストを追加. 既存 phase_times テストも class 単位で整理. (NA.)
- `IDetectionPipeline.pipeline_mode` プロパティの cpu / gpu 反映を RTDetr / SSD 両方で検証するテストを追加. (NA.)
- `resolve_pipeline_mode()` の入力組合せを `@pytest.mark.parametrize` で網羅. ONNX + gpu 拒否時の `ValueError` メッセージも検証. (NA.)

### Fixed
- 無し

### Removed
- 無し

## v0.16.0 (2026-04-18)

### Added
- WebAPI 基盤を追加. `pochi serve -m <model_path>` で FastAPI + uvicorn による推論 API サーバーを起動可能. ([#439](https://github.com/kurorosu/pochidetection/pull/439))
  - `pochidetection/api/` モジュール (`app.py`, `config.py`, `schemas.py`, `routers/health.py`) を新規作成.
  - メタ情報 4 エンドポイント: `GET /api/v1/health`, `/version`, `/model-info`, `/backends` を実装.
  - 起動時 lifespan で `resolve_and_setup_pipeline` 経由でモデルロード + ダミー画像で warmup 推論を実行.
  - `fastapi>=0.135.0`, `uvicorn[standard]>=0.43.0` を依存に追加.
  - 推論バックエンド抽象 (`IDetectionBackend`) と `/detect` エンドポイントは Issue #436 で追加予定.
- 検出エンドポイント `POST /api/v1/detect` を追加. base64 エンコードされた画像 (raw / jpeg) を受け取り, 検出結果 (bbox, class, confidence) を返却. ([#440](https://github.com/kurorosu/pochidetection/pull/440))
  - `IDetectionBackend` 抽象と 3 backend (`PyTorchDetectionBackend`, `OnnxDetectionBackend`, `TrtDetectionBackend`) を `api/backends.py` に実装. 既存 `IDetectionPipeline` を内部でラップ.
  - `api/serializers.py` に `RawArraySerializer` / `JpegSerializer` を追加.
  - `DetectRequest` (score_threshold 対応) / `DetectionDict` (bbox `[x1,y1,x2,y2]`) / `DetectResponse` を `api/schemas.py` に追加.
  - `#435` で導入した `ModelHolder` を `IDetectionBackend` に置換. `/model-info`, `/backends` は engine 参照に書き換え.
  - `docs/api-server.md` に raw / jpeg リクエスト例, レスポンス形式, エラーコード一覧, バックエンド自動判定の仕様を追記. README.md にも `pochi serve` の概要とエンドポイント一覧を追加.

### Changed
- `POST /api/v1/detect` にフェーズ別タイミング計測を追加. ボトルネック特定用の観測機能. ([#448](https://github.com/kurorosu/pochidetection/pull/448))
  - `IImageSerializer.deserialize()` / `IDetectionBackend.predict()` の戻り値を `(result, phase_times)` タプルに変更.
  - `DetectResponse` に optional な `phase_times_ms: dict[str, float]` フィールドを追加. `b64_decode_ms`, `imdecode_ms` (jpeg のみ), `reshape_ms` (raw のみ), `cvt_color_ms`, `pipeline_preprocess_ms`, `pipeline_inference_ms`, `pipeline_postprocess_ms` を出力.
  - router 側で `model_dump_ms` を含む各フェーズを `time.perf_counter()` で計測し, ログとレスポンス両方に出力.
- `docs/api_detect_inference_variance_investigation.md` の真因仮説を "PyTorch caching allocator" から "asyncio / uvicorn / Windows timer 領域" に更新. カメラストリームでの 14ms 安定と Web 調査を新エビデンスとして追記. ([#450](https://github.com/kurorosu/pochidetection/pull/450))
- `POST /api/v1/detect` の inference フェーズに CUDA Event 計測 (`pipeline_inference_gpu_ms`) と リクエスト間隔 (`gap_since_last_request_ms`) を追加. 真因が NVIDIA driver の adaptive clock policy であることを `nvidia-smi --lock-gpu-clocks` 検証で確定. 資料 `docs/api_detect_inference_variance_investigation.md` を最終結論で改訂. ([#452](https://github.com/kurorosu/pochidetection/pull/452))
- `POST /api/v1/detect` の INFO ログを 1 行サマリに圧縮し可読性を改善. GPU クロック (pynvml 経由, `pochidetection/api/gpu_clock.py` 新設) もログ末尾に併記. ([#455](https://github.com/kurorosu/pochidetection/pull/455))
  - `DetectResponse.phase_times_ms` を pipeline 内訳 4 値に縮小し, serializer / backend の breakdown 計測を撤去.
- `RTDetrPipeline` / `SsdPipeline` に GPU 上 preprocess 経路を追加し起動時オプション `--pipeline cpu/gpu` (default: `gpu`) で切替可能化. preprocess 7-12ms → 3-4ms に短縮し CLI / API / カメラ全経路で効果. ONNX backend は CPU 経路のみ対応 (gpu 明示時は起動拒否). ログ末尾に `pipeline=cpu/gpu` を併記. ([#456](https://github.com/kurorosu/pochidetection/pull/456))
  - GPU 経路: numpy → uint8 tensor (CHW) → CPU 上で uint8 のまま resize → GPU バッファに `copy_` で float32 化 + H2D → `div_(255)` で `[0,1]` 化. バッファは shape mismatch 時のみ再確保.
  - `IDetectionPipeline.pipeline_mode` プロパティを追加 (router の INFO ログ出力用).
  - `DetectionConfigDict` / `DetectionConfig` / `ServerConfig` に `pipeline_mode` (CLI 引数の上書き対象) を追加.

### Fixed
- 無し

### Removed
- 無し

## Archived Changelogs

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
