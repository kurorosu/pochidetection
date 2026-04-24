# テストガイド

pochidetection のテスト方針と, 実際のリポジトリで使われている慣例をまとめた開発者向けガイド.

関連ドキュメント:

- `.claude/rules/testing.md` — グローバルなテスト方針 (本ガイドの出発点)
- `tests/docs/slow_tests.md` — slow マーカー付きテストの一覧

## テスト哲学

- **classical test を優先する**. 可能な限り mock を避け, 実オブジェクトを使って実挙動を検証する.
- **実結果を検証する**. 内部呼び出しの spy ではなく, 入力 → 出力の振る舞いを assert する.
- **ファイル I/O は `tmp_path` を使う**. モジュールレベルのディレクトリや cwd 相対パスに依存しない.
- **動的な値にはパターンマッチを使う**. 日時やパスなど環境依存値は正規表現 (`match=` / `re.search`) で検証し, mock で固定化しない.
- **mock は例外的手段**. 外部プロセス (subprocess), 外部ハードウェア (NVML 経由の GPU クロック) など, 実呼び出しがテスト環境で不安定 / 不可能な場合に限定する.

## ディレクトリ構成

テストは `tests/` 配下を `pochidetection/` のサブパッケージに概ね対応させて配置する.

```
tests/
├── __init__.py
├── conftest.py              # ルート共通 fixture (モデル / COCO アノテ / 予測結果)
├── docs/                    # テストガイド / slow マーカー一覧
├── test_api/                # WebAPI (FastAPI) のテスト + ローカル conftest
├── test_cli/                # CLI エントリポイントのテスト
├── test_configs/            # Config スキーマ検証
├── test_core/               # Collate, preprocess など基盤モジュール
├── test_datasets/           # COCO データセット / augmentation
├── test_inference/          # 推論バックエンド (PyTorch)
├── test_interfaces/         # 抽象基底クラスの契約テスト
├── test_logging/
├── test_models/             # RTDetr / SSD300 / SSDLite
├── test_onnx/               # ONNX エクスポート / バックエンド (slow)
├── test_pipelines/          # 学習 / 推論パイプライン構築
├── test_reporting/
├── test_scripts/
├── test_tensorrt/           # TensorRT エクスポート / 推論 (slow + importorskip)
├── test_training/
├── test_utils/
└── test_visualization/
```

命名規則:

- テストファイルは `test_<module>.py` (対象モジュール名と揃える)
- テストクラスは `Test<ClassOrFeature>` (classical な振る舞いごとにグルーピング)
- テスト関数は `test_<振る舞い>_<条件>` の形で, docstring に期待挙動を日本語で 1 行書く

## Fixtures

### 配置

- **`tests/conftest.py`**: プロジェクト全体で共有する重量級 fixture を置く. 現状は `rtdetr_model` / `ssd300_model` / `ssdlite_model` (`scope="session"`) と `coco_annotation` / `sample_predictions` を提供.
- **`tests/<subdir>/conftest.py`**: そのサブディレクトリに閉じた fixture を置く. 例:
  - `tests/test_api/conftest.py` — `reset_engine` (`autouse=True`) で各テスト後に `state._engine` を復元
  - `tests/test_tensorrt/conftest.py` — `pytest.importorskip("tensorrt")` とダミー ONNX / エンジン / キャリブレータ生成 (`scope="session"`)

### スコープ選定

| スコープ | 使いどころ | 例 |
|---------|----------|----|
| `session` | 初期化が重い (モデル / ONNX エクスポート / TensorRT エンジン構築) | `rtdetr_model`, `dummy_onnx_path`, `engine_path` |
| `class` | 同一テストクラス内で再利用できる軽量データ | `single_epoch_history` |
| 関数 (default) | ファイル I/O を伴い各テストで新規ディレクトリが必要 | `coco_annotation`, `sample_predictions` |

### 命名

- 生成されるオブジェクト名を素直に付ける (`rtdetr_model`, `coco_annotation`, `engine_path`).
- 予測 / アノテのペアは対応がわかる名前で揃える (`coco_annotation` / `sample_predictions` は同じ画像 ID 空間を共有).
- `scope="session"` のモデル fixture は `eval()` 済みで返す. 学習モードが必要なテストでは呼び出し側で `was_training = model.training` → `model.train()` → 復元パターンを取る (session 共有のため grade cleanup が重要).

### session fixture と conflict するパラメータ

session fixture と異なる `num_classes` などが必要なテストはテスト内で個別初期化するが, その場合は **`@pytest.mark.slow`** を付ける (初期化コストが大きいため). 詳細は `tests/docs/slow_tests.md`.

## Parametrize の活用

### 使いどころ

- **境界値テスト**: `ge=0.0, le=1.0` のような範囲制約を `[0.0, 0.5, 1.0]` / `[-0.01, 1.01, math.inf]` の 2 本に分けて parametrize する (`tests/test_api/test_schemas.py`).
- **同一ロジックを複数デバイス / モードで回す**: `device = ["cpu", "cuda"]`, `mode = ["cpu", "gpu"]` のようなケース (`tests/test_core/test_preprocess.py`, `tests/test_pipelines/test_pipeline_mode.py`).
- **設定の組み合わせ爆発を抑える**: 同じ検証を異なる入力で繰り返すときは `@pytest.mark.parametrize` で 1 関数にまとめる.

### 書き方

```python
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_accepts_in_range(self, value: float) -> None:
    """範囲内 (0.0, 境界, 1.0) は受理."""
    ...
```

条件付きスキップと組み合わせる場合は `pytest.param(..., marks=...)` を使う:

```python
_DEVICE_PARAMS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
        ),
    ),
]

@pytest.mark.parametrize("device", _DEVICE_PARAMS)
def test_returns_float32_in_unit_interval(self, device: str) -> None:
    ...
```

### ID 指定

- 値が短い primitive (数値 / bool / 短い文字列) は pytest が自動生成する ID で十分.
- 複雑なオブジェクトを渡す場合は `ids=[...]` で人間可読な ID を明示する.

## tmp_path の活用

ファイル I/O を伴うテストは `tmp_path` / `tmp_path_factory` を必ず使う.

### 関数スコープの `tmp_path`

テスト関数ごとに独立した一時ディレクトリ. fixture からファイルを書き出す典型例:

```python
@pytest.fixture()
def coco_annotation(tmp_path: Path) -> Path:
    """テスト用 COCO アノテーション."""
    ann = {...}
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(ann), encoding="utf-8")
    return ann_path
```

### session スコープの `tmp_path_factory`

セッション中 1 回だけ生成して共有する重量級データ (ONNX / TensorRT エンジン / キャリブレーション画像) は `tmp_path_factory.mktemp(...)` を使う:

```python
@pytest.fixture(scope="session")
def dummy_onnx_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_dir = tmp_path_factory.mktemp("trt_onnx")
    output_path = tmp_dir / "tiny_model.onnx"
    torch.onnx.export(...)
    return output_path
```

### ワークスペース擬似構造のテスト

CLI の `work_dirs/` 解決のように階層ディレクトリを期待するテストは, `tmp_path` 配下に同じ構造を再現する (`tests/test_cli/test_cli_integration.py`):

```python
workspace = tmp_path / "work_dirs" / "20260314_001"
workspace.mkdir(parents=True)
```

## slow marker

### 設定

`pyproject.toml` で `addopts = "-v -n 6 -m 'not slow'"` となっており, **デフォルトの `uv run pytest` では slow マーカー付きテストは実行されない**.

```toml
[tool.pytest.ini_options]
addopts = "-v -n 6 -m 'not slow'"
markers = ["slow: 時間のかかるテスト (uv run pytest -m slow で実行)"]
```

### 実行方法

```bash
uv run pytest               # 通常 (slow 除外, デフォルト)
uv run pytest -m slow       # slow のみ
uv run pytest -m ""         # 全テスト (slow 含む)
```

### 付与基準

以下のいずれかに該当する場合に `@pytest.mark.slow` を付ける:

- **実モデルの新規初期化**: `RTDetrModel`, `SSD300Model`, `SSDLiteModel` の session fixture と異なるパラメータ (`num_classes=4` 等) で再初期化するケース (0.7s〜1.5s/テスト)
- **ONNX エクスポート**: 実モデルからの ONNX エクスポートやセッション作成 (0.5s〜0.8s/テスト)
- **TensorRT エンジンビルド**: GPU で engine をビルドするケース (2s〜3s/テスト)
- **subprocess 起動**: CLI のサブプロセステスト (3s〜7s/テスト)

付与単位:

- **ファイル全体**: ファイル内のすべてのテストが slow なら先頭に `pytestmark = pytest.mark.slow` を置く (例: `tests/test_onnx/test_rtdetr_exporter.py`, `tests/test_tensorrt/test_exporter.py`, `tests/test_cli/test_cli.py`).
- **個別テスト**: 一部だけが slow なら `@pytest.mark.slow` を個別メソッドに付ける (例: `tests/test_models/test_rtdetr_model.py` の `test_custom_num_classes`).

現状の slow テスト一覧と付与理由は `tests/docs/slow_tests.md` を参照.

## CUDA / GPU テスト

CUDA 必須のテストは `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` でスキップさせる. CPU 環境 / CI でもテストが落ちないようにする.

### 単体テスト

```python
import torch
import pytest

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_returns_real_clock_mhz_on_cuda_environment() -> None:
    """CUDA 環境では実際の GPU graphics clock (MHz) を整数で返す."""
    result = gpu_metrics.get_gpu_clock_mhz()
    assert result is not None
    assert isinstance(result, int)
```

### parametrize と組み合わせる

CPU / CUDA 両方で同じ振る舞いを検証するときは `pytest.param` で CUDA 側にだけ skipif を付ける (前述 `_DEVICE_PARAMS` パターン). これにより CPU 環境では CPU ケースのみ実行され, CUDA 環境では両方走る.

### ネイティブ依存のスキップ

TensorRT のようにインストール自体が環境依存のライブラリは, モジュール冒頭で `pytest.importorskip` を使う:

```python
import pytest
pytest.importorskip("tensorrt")

from pochidetection.tensorrt import TensorRTExporter
```

`tests/test_tensorrt/` 配下と `tests/test_onnx/conftest.py` でこのパターンを採用している.

### GPU 状態の共有に注意

TensorRT / CUDA はプロセス横断で plugin キャッシュ / GPU コンテキストを共有する. pytest を worktree 並列で走らせる場合は競合するため, サブエージェントは `SKIP=pytest` で commit し, pytest はメインエージェントが直列実行する (`.claude/rules/worktree-workflow.md`).

## テストデータ

- 静的なフィクスチャファイルを置く `tests/fixtures/` ディレクトリは**現状存在しない**. COCO アノテーションや予測結果は `tests/conftest.py` の fixture 内で dict / JSON を生成する方針を取っている.
- 画像データが必要なケース (TensorRT キャリブレータのテスト等) は, fixture 内で `PIL.Image.new` でダミー画像を生成し `tmp_path_factory` 配下に保存する (`tests/test_tensorrt/conftest.py` の `calib_image_dir`).
- この方針のメリット:
  - fixture が self-contained になり, テストファイル単体で入力データの内容を追える
  - バイナリをリポジトリに含めないため, diff が読みやすい
- 将来的に大きなテスト画像 / アノテーションが必要になった場合は `tests/fixtures/` 新設を検討するが, 現時点では conftest での動的生成を優先する.
