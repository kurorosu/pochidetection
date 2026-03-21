# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 画像推論で検出ボックスのクロップ画像を `inference_XXX/crop/` に自動保存する機能を追加 (デフォルト有効). `--no-crop` で無効化可能. ([#396](https://github.com/kurorosu/pochidetection/pull/396))
  - `InferFn` の型定義を `Callable` から `Protocol` に変更し, keyword-only 引数 (`save_crop`) に対応.
- `torchvision.transforms.v2` ベースの Data Augmentation パイプラインを導入. config.py の `augmentation` セクションで変換を設定可能. 学習データのみに適用, bbox は `tv_tensors.BoundingBoxes` で同時変換. N/A.
  - `docs/augmentation.md` に設定方法・変換一覧・使用例を記載.

### Changed

### Fixed
- 無し

### Removed
- 無し

## v0.13.0 (2026-03-21)

### Added
- Webcam 推論中に `s` キーで Windows カメラ設定ダイアログ (DirectShow) を表示する機能を追加. `DisplaySink` に `cap` パラメータを追加し, `StreamReader.cap` プロパティ経由で `VideoCapture` を参照. ([#386](https://github.com/kurorosu/pochidetection/pull/386))
- リアルタイム推論完了時に, 使用した config ファイルのコピー, カメラプロパティ, 実測 E2E FPS サマリーを推論フォルダ (`stream_metadata.json`) に保存する機能を追加. ([#387](https://github.com/kurorosu/pochidetection/pull/387))
  - `process_frames()` の戻り値を `FrameProcessingResult` dataclass に変更.
  - `cli/commands/infer.py` の遅延インポートを全てトップレベルに移動.
- リアルタイム推論の FPS オーバーレイに全フェーズ別内訳 (capture/pre/infer/post/draw/display) を縦書き表示 (白縁取り + 黒文字). サマリーログと `stream_metadata.json` にもフェーズ別平均を出力. 既存の `PhasedTimer` を再利用. ([#388](https://github.com/kurorosu/pochidetection/pull/388))
- config.py から `camera_fps` と `camera_resolution` を設定可能にする機能を追加. `StreamReader.apply_camera_settings()` でカメラに適用し, 設定値と実際の値が異なる場合は警告ログを出力. ([#392](https://github.com/kurorosu/pochidetection/pull/392))

### Changed
- `--record` オプションをパス指定からフラグに変更し, 録画ファイルを推論フォルダ (`inference_XXX/recording.mp4`) に自動保存するように改善. ([#394](https://github.com/kurorosu/pochidetection/pull/394))
- リアルタイム推論のフレーム処理から PIL 変換を除去し, パフォーマンスを改善. ([#389](https://github.com/kurorosu/pochidetection/pull/389))
  - `IDetectionPipeline.run()` が `Image.Image | np.ndarray` を受け付けるように拡張.
  - `Visualizer.draw_cv2()` を追加し, OpenCV で BGR フレームに直接描画.
  - `process_frames()` から `Image.fromarray()` / `np.array()` / RGB→BGR 変換を除去.
  - FPS オーバーレイを E2E FPS (capture + 推論 + 描画 + display 全込み) に変更.

### Fixed
- 無し

### Removed
- 無し

## Archived Changelogs

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
