# `POST /api/v1/detect` 推論時間の振れ調査

関連 Issue: #446 (フェーズ別計測), #447 (バッファ再利用による解決)

## 概要

`POST /api/v1/detect` の `pipeline_inference_ms` が CLI (`pochi infer`) の **8.4ms 安定** に対して **40ms 〜 100ms で大きく振れる** 問題を調査した. フェーズ別タイミング計測を追加し, スレッド ID, GPU クロック, async/sync, TRT context 紐付け等の主要仮説を 1 つずつ棄却した上で, 姉妹プロジェクト pochitrain の Issue #391 / PR #440 と突き合わせて **PyTorch caching allocator のメモリブロック再編成** が原因と特定した. 解決方針は **GPU 入力バッファの事前確保 + in-place 再利用 + 複数回 warmup** で, 別 Issue #447 として実装する.

## 検証環境

- OS: Windows 11 Home
- GPU: NVIDIA GeForce (ノート PC)
- Python 3.13
- pochidetection PR #440 (検出エンドポイント) マージ後の dev ブランチ
- モデル: RT-DETR (TensorRT engine, 640×640)
- テストデータ: `data/val/JPEGImages/` の 36 枚 (512×512 〜 一般撮影サイズ)

## 1. 現象

PR #440 のマージ後に実機検証で発見:

| 計測対象 | E2E | 推論本体 (`pipeline_inference_ms`) |
|---|---|---|
| CLI (`pochi infer`) | 12.3ms | **8.4ms 安定** |
| API (`POST /detect`) | 49 〜 90ms | **38 〜 130ms 振動** |

API は CLI 同等のパスを通っている (TRT engine 同一, 同じ `pipeline.run()`) はずなのに **5 〜 13 倍遅く, 値が安定しない**.

## 2. 計測機能の追加

支配要因を特定するため, `/detect` の各フェーズを個別計測する仕組みを追加した (PR で実装).

```
1. base64.b64decode             (serializers.py)
2. cv2.imdecode (jpeg のみ)      (serializers.py)
3. request.model_dump()         (routers/inference.py)
4. cv2.cvtColor BGR→RGB         (backends.py)
5. Image.fromarray              (rtdetr_pipeline.py)
6. v2.Compose (Resize+ToTensor) (rtdetr_pipeline.py)
7. inference (TRT execute)      (rtdetr_pipeline.py)
8. postprocess (HF + NMS)       (rtdetr_pipeline.py)
```

`DetectResponse.phase_times_ms` に dict として返却し, 同時に INFO ログにも出力.

## 3. 仮説の棄却プロセス

### 仮説 A: `Image.fromarray` の PIL コピー (numpy → PIL)

**根拠**: CLI は `Image.open()` で PIL 画像を直接 pipeline に渡し fromarray をスキップ. API は numpy → PIL のフル解像度コピーを毎回実行.

**棄却**: 実測で `pipeline_preprocess_ms` (fromarray 込み) は 8.5ms 程度. 全体差 (~80ms) を説明できない.

### 仮説 B: GPU クロック P-state ダウン

**根拠**: HTTP リクエスト間の idle gap (10-30ms) で GPU がダウンクロックし, 次リクエストでランプアップ時間がかかる.

**検証**: スタンドアロンスクリプトで `time.sleep(0.030)` を 30 イテレーション挟んで実行.

**棄却**: 8ms 安定のまま. GPU クロック仮説は CLI が連続実行で保たれる事実と整合しない (CLI も画像 I/O で同等の gap がある).

### 仮説 C: `async def` ハンドラと event loop スレッド

**根拠**: FastAPI の async ハンドラは event loop スレッドで実行. CUDA 操作との相性問題?

**検証**: ハンドラを `def` (sync) に変更 → FastAPI は anyio thread pool worker で実行.

**棄却**: 同じく 38-130ms の bimodal 振動. async/sync の差異ではない.

### 仮説 D: TRT IExecutionContext のスレッド紐付け

**根拠**: `def` 化で thread pool worker スレッド (tid=16292) で predict が走るが, `build_engine` は main スレッド (tid=6564). TRT context のスレッド紐付けで性能劣化?

**検証**: `async def` に戻して thread ID をログ出力.

**棄却**: `async def` では `build_engine` と `predict` 両方とも tid=23180 (同一 main スレッド). スレッド一致時も振れあり.

### 仮説 E: スタンドアロンとの最後の差分

スタンドアロンスクリプト (`build_engine` + tight loop で `predict`) は 8ms 張り付き. つまり **`backends.predict()` のコード自体は正常**.

```
=== tight loop: 30 iterations ===
predict_total_ms          mean=13.40  median=13.56  min=12.00  max=14.62
pipeline_inference_ms     mean= 8.42  median= 8.40  min= 8.05  max= 8.90
```

API でも **最初の 3-4 リクエストは 8ms** で出る. その後急に劣化:

```
req#  e2e   inference   note
  1   74.8  7.94       (warmup, postprocess=51ms)
  2   25.4  8.24       ← CLI 速度!
  3   60.4  45.32      ← 突然遅くなる
  ...
 18   96.0  80.77
 19  121.4 104.83      ← ピーク
 25   59.7  45.49      ← 中速モードに復帰
 31  107.8  91.17      ← 再び遅い
 34   57.2  42.83      ← 中速
```

**「8ms (CLI 速度) → 中速 (40-50ms) ⇔ 低速 (90-100ms)」** の 3 段階遷移. 初期だけ CLI と同速で, 以降は 40ms と 90ms の bimodal を行き来.

## 4. 真因: PyTorch caching allocator のブロック再編成

姉妹プロジェクト pochitrain の Issue [#391](https://github.com/kurorosu/pochitrain/issues/391) と PR [#440](https://github.com/kurorosu/pochitrain/pull/440) で同様の振れが報告されており, **対策としてメモリ事前確保 + in-place 演算で解消** した実績がある.

### 振れの仕組み

`pipeline.preprocess()` の各リクエストで以下の中間 tensor が生成 → 破棄される:

```python
# rtdetr_pipeline.py:85-88
pixel_values = self._transform(image).unsqueeze(0).to(self._device)
```

- `_transform(image)` の戻り値: `(3, 640, 640) float32` を新規確保
- `.unsqueeze(0)`: `(1, 3, 640, 640)` view
- `.to(self._device)`: GPU 上に新規 `(1, 3, 640, 640) float32` (~3.1MB) を確保

これがリクエスト毎に発生するため, PyTorch の caching allocator が **ある頻度で内部ブロックを再配置** する. 再配置時に数 ms 〜 10ms 超のスパイクが入り, それが累積して 40-100ms の振動になる.

**スタンドアロンが速い理由**: tight loop の中で同じパターンの確保/解放を高速に繰り返すため, allocator の内部キャッシュがすぐ「定常状態」に到達してブロック再配置がほぼ起きない. 一方 API は HTTP 応答送信や Pydantic 検証で間に CPU 処理が挟まり, allocator のヒューリスティックが「リクエスト間で再編成すべき」と判断する瞬間が発生する.

### CLI が速い理由

CLI (`pochi infer`) は `pipeline.run()` を tight loop で連続実行する (画像保存なども CPU 主体で短時間). スタンドアロンと同様に caching allocator が定常化する.

## 5. 解決策: pochitrain #440 方式

```python
# IDetectionBackend.predict() 内の擬似コード
tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
if self._input_buf is None or self._input_buf.shape != tensor.shape:
    self._input_buf = torch.empty(
        tensor.shape, dtype=torch.float32, device=self._device
    )
# 起動後初回だけ allocator に確保要求, 以降は再利用
self._input_buf.copy_(tensor)               # uint8→float32 dtype 変換も兼ねる
self._input_buf.sub_(self._mean_255).div_(self._std_255)  # in-place 正規化
```

**効果 (pochitrain 実績)**:
- 振れ範囲: 0.5 〜 16ms → **p99 < 5ms** に収束
- ピーク最速値はやや悪化 (allocator スパイクの代わりに in-place 演算オーバーヘッド)
- 平均と p99 が大幅に改善, 運用上の安定性が向上

**追加対策**:
- `_WARMUP_ITERATIONS = 3` (現状 1 回) で起動時に 3 回 dummy 推論. TRT autotuner / cuDNN JIT を確実に warm-up.
- `np.zeros` → `np.random.randint` で warmup. 実データに近い分布でカーネル選択.

詳細は **Issue #447** で別途実装.

## 6. 棄却された仮説の整理

| 仮説 | 検証方法 | 結果 |
|---|---|---|
| `Image.fromarray` PIL コピー | preprocess フェーズ単独計測 | ❌ 8.5ms で全体振れを説明できず |
| GPU クロック P-state | standalone + `sleep(0.030)` × 30 | ❌ 8ms 安定維持 |
| async vs sync def (event loop) | ハンドラ書き換え + 再ベンチ | ❌ どちらも bimodal |
| TRT context スレッド紐付け | `threading.get_ident()` ログ | ❌ async は同一スレッド |
| **caching allocator 再編成** | pochitrain #391/#440 と突合 | ✅ 原因確定 |

## 7. 教訓と再発防止

### Caching allocator の振れは tight loop では再現しない

ベンチを「ループでひたすら呼ぶ」だけだと allocator が定常化して問題が消える. 実運用に近い **「リクエスト間に CPU 処理 (HTTP 応答 / serialize / 別タスク) が挟まる」状況** で初めて観測される.

### フェーズ別計測の重要性

`e2e_time_ms` だけ見ていた段階では「preprocess が遅い」「PIL コピーが原因」と誤った仮説に走った. **PhasedTimer + 後段のフェーズ別ログ** を入れて初めて「preprocess は問題なく, inference 本体が振れている」と特定できた. PR #440 マージ時点で `phase_times_ms` を入れておけば, ここまでの調査時間が短縮できた.

### 姉妹プロジェクトの過去事例を先に当たる

pochitrain の Issue #391 / PR #440 を先に確認していれば, 仮説 A-D の検証は不要だった. 同じ pochi_series 系プロジェクト間で **共通の API 設計上の落とし穴** は再発しやすい. 今後の API 系新機能では pochitrain 側の対応 PR / Issue を最初に確認するワークフローを推奨.

### 案 B/C/pochitrain 方式の選択

事前検討では 3 案 (案 B: pipeline numpy バイパス / 案 C: serializer の PIL 化 / pochitrain GPU 方式) を並列に評価していたが, **真因が caching allocator にあると判明した時点で答えは pochitrain 方式一択**. 案 B/C は preprocess の PIL コピーを削るだけで, 真の振れ要因 (GPU メモリ確保) は解消しない.

## 8. 関連リンク

- pochidetection PR #440: 検出エンドポイント `/detect` 追加
- pochidetection Issue #446: フェーズ別計測の追加
- pochidetection Issue #447: GPU 入力バッファ再利用 (本問題の解決)
- pochitrain Issue #391: 同問題の原因分析 (https://github.com/kurorosu/pochitrain/issues/391)
- pochitrain PR #440: warmup + GPU 入力バッファ事前確保 (https://github.com/kurorosu/pochitrain/pull/440)
