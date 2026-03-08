# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- `scripts/common/training.py` の `build_early_stopping` と `scripts/common/inference.py` の `resolve_model_path`, `collect_image_files` のユニットテストを追加. ([#202](https://github.com/kurorosu/pochidetection/pull/202).)

### Changed
- `LoggerManager._loggers` をクラス変数からインスタンス変数に変更し, テスト時のシングルトンリセットを容易にした. ([#192](https://github.com/kurorosu/pochidetection/pull/192).)
- `_build_data_loaders` を `rtdetr/train.py` と `ssdlite/train.py` から `scripts/common/training.py` の `build_data_loaders` に共通化. データセット生成はファクトリ関数で注入. ([#193](https://github.com/kurorosu/pochidetection/pull/193).)
- `Visualizer`, `InferenceSaver`, `DetectionSummary`, `DetectionResultRow` 等の推論共通コンポーネントを `scripts/rtdetr/inference/` から `scripts/common/` に移動. `ssdlite/infer.py` が `rtdetr` パッケージに依存しなくなった. ([#195](https://github.com/kurorosu/pochidetection/pull/195).)

### Fixed
- `test_map_evaluator.py` から `test_coco_utils.py` と重複する `TestExtractBasename`, `TestXywhToXyxy` を削除. (N/A.)
- `cli/main.py` で `export_onnx` / `export_trt` がトップレベル import されていたのを lazy import に変更. TensorRT 未インストール環境で `pochi train` / `pochi infer` 実行時に不要なモジュール読み込みが発生しなくなった. ([#196](https://github.com/kurorosu/pochidetection/pull/196).)
- `RTDetrModel.load()` で `_num_classes` が更新されず, load 前のクラス数が残る問題を修正. ([#197](https://github.com/kurorosu/pochidetection/pull/197).)
- `Visualizer` のフォールバックフォントで `ImageFont.load_default()` が `size` 引数なしで呼ばれ, Linux/Docker 環境でラベルが極小になる問題を修正. `load_default(size=font_size)` に変更. ([#198](https://github.com/kurorosu/pochidetection/pull/198).)
- `InferenceSaver` で `base_dir` が存在しない場合に `FileNotFoundError` が発生する問題を修正. ディレクトリを自動作成するよう変更. ([#199](https://github.com/kurorosu/pochidetection/pull/199).)
- `IDetectionDataset.__getitem__` の docstring が実装と不一致だったのを修正. 返り値を `pixel_values` と `labels` に更新. ([#200](https://github.com/kurorosu/pochidetection/pull/200).)
- `IDetectionModel.forward` の docstring が `RTDetrModel` の実装と不一致だったのを修正. モデル固有の返り値 (RT-DETR: `pred_logits`/`pred_boxes`, SSDLite: `predictions`) を明記. ([#201](https://github.com/kurorosu/pochidetection/pull/201).)
- SSDLite の `_validate` で `model.train()` に切り替えた際に BatchNorm の `running_mean` / `running_var` が検証データで更新される問題を修正. BN 統計を退避・復元する方式で保護. ([#191](https://github.com/kurorosu/pochidetection/pull/191).)

### Removed
- 無し.

## v0.6.2 (2026-03-07)

### Added
- 無し.

### Changed
- 無し.

### Fixed
- `DetectionConfig` で SSDLite が無視する設定項目 (`model_name`, `pretrained`, `nms_iou_threshold`) にデフォルト以外の値を指定した場合に `UserWarning` を発行するバリデーションを追加. サンプル設定 `ssdlite_coco.py` から該当項目を削除. ([#168](https://github.com/kurorosu/pochidetection/pull/168).)
- SSDLite の label オフセット (+1/-1) を `SSDLiteModel.forward()` に集約し, `SsdCocoDataset` と `train.py _validate()` から分散していたオフセット処理を除去. 推論時の背景クラス予測 (label=-1) を除去するガードを追加. ([#169](https://github.com/kurorosu/pochidetection/pull/169).)
- SSDLite の前処理に `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` を追加. `SsdCocoDataset` (学習) と `ssdlite/infer.py` (推論) の両方に適用し, MobileNetV3 バックボーンが期待する ImageNet 正規化を実施. ([#171](https://github.com/kurorosu/pochidetection/pull/171).)

### Removed
- 無し.

## Archived Changelogs

- [0.6.x](changelogs/0.6.x.md)
- [0.5.x](changelogs/0.5.x.md)
- [0.4.x](changelogs/0.4.x.md)
- [0.3.x](changelogs/0.3.x.md)
- [0.2.x](changelogs/0.2.x.md)
- [0.1.x](changelogs/0.1.x.md)

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
