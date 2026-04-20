# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 推論時 preprocess 後画像のデバッグ保存を追加. top-level `infer_debug_save_count` (既定 1) で制御し, CLI / 動画 / カメラ / WebAPI の全経路で先頭 N 枚を `{output_dir}/infer_debug/` へ保存. (NA.)

### Changed
- Pipeline の letterbox 幾何パラメータ (`_last_letterbox_params`) をインスタンス属性から preprocess 戻り値経由の request-scoped 受け渡しに変更し, 同一 pipeline を複数 thread から並行呼出しても bbox 逆変換が混線しないようにした. ([#576](https://github.com/kurorosu/pochidetection/pull/576))

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
