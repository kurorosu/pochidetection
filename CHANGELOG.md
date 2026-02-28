# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- ONNX エクスポート機能 (`OnnxExporter`) を追加. `export`, `verify`, `load_model` メソッドで学習済み RT-DETR モデルの ONNX 変換・検証・復元に対応. CLI `pochidet-rtdetr export` コマンドも追加. (N/A.)

### Changed
- ConfigLoader を Pydantic スキーマベースに刷新し, `pydantic` 依存を追加. 設定値は `ValidationError` で詳細に検証し, 未知キーも拒否する形に変更した. (N/A.)
- plotter 系テストで class スコープ fixture と tmp_path_factory を導入し, HTML 生成を共有して `fig.write_html()` 呼び出し回数を削減. あわせて `test_rtdetr_model.py` の forward 入力を軽量化し, 実挙動検証を維持したまま実行時間を短縮. ([#38](https://github.com/kurorosu/pochidetection/pull/38))
- テストコードを古典派テストに移行 (MagicMock 除去, プライベート属性アクセス除去, プレースホルダーテスト実装) し, テストディレクトリをモジュール構成に合わせて整理した. ([#35](https://github.com/kurorosu/pochidetection/pull/35))
- `test_rtdetr_model.py` の `model` / `model_for_training` fixture を `scope="class"` に変更し, setup 時間を約 5.1s から約 1.0s に削減. ([#36](https://github.com/kurorosu/pochidetection/pull/36))

### Fixed
- なし.

### Removed
- なし.

## v0.1.0 (2026-02-28)

### 概要
- pochidetection の初期リリース. RT-DETR による学習・推論パイプラインを実装.

### Added
- インターフェース定義 (`IDetectionModel`, `IDetectionDataset`, `IPlotter`) ([#2](https://github.com/kurorosu/pochidetection/pull/2))
- COCO フォーマットデータセット (`CocoDetectionDataset`) ([#3](https://github.com/kurorosu/pochidetection/pull/3))
- RT-DETR モデル (`RTDetrModel`) - HuggingFace Transformers ラッパー ([#4](https://github.com/kurorosu/pochidetection/pull/4))
- 損失関数と評価指標の実装 ([#5](https://github.com/kurorosu/pochidetection/pull/5))
- Factory/Config/Utils の実装 ([#6](https://github.com/kurorosu/pochidetection/pull/6))
- Trainer の実装 ([#7](https://github.com/kurorosu/pochidetection/pull/7))
- CLI エントリポイント (`pochidet-rtdetr train` / `pochidet-rtdetr infer`) ([#8](https://github.com/kurorosu/pochidetection/pull/8))
- 作業ディレクトリ自動生成機能 (`WorkspaceManager`) ([#12](https://github.com/kurorosu/pochidetection/pull/12))
- 画像サイズ設定の追加 ([#18](https://github.com/kurorosu/pochidetection/pull/18))
- フォルダ一括推論機能 ([#20](https://github.com/kurorosu/pochidetection/pull/20))
- クラスラベル名と色の対応機能 (`ColorPalette`, `LabelMapper`) ([#22](https://github.com/kurorosu/pochidetection/pull/22))
- 推論時間の計測・集計機能 (`InferenceTimer`) ([#24](https://github.com/kurorosu/pochidetection/pull/24))
- 推論高速化オプション (cudnn_benchmark, FP16) ([#28](https://github.com/kurorosu/pochidetection/pull/28))
- 学習可視化 - Loss 曲線, mAP 曲線, PR 曲線の HTML 出力 (Plotly) ([#31](https://github.com/kurorosu/pochidetection/pull/31))

### Changed
- CLI とオーケストレーションを分離, 未使用クラス・インターフェースを削除 ([#15](https://github.com/kurorosu/pochidetection/pull/15))
- `InferenceTimer` に Context Manager と `reset()` を追加 ([#26](https://github.com/kurorosu/pochidetection/pull/26))

### Fixed
- なし.

### Removed
- なし.

## Archived Changelogs

なし.

運用ルールは [`changelogs/README.md`](./changelogs/README.md) を参照してください.
