# `POST /api/v1/detect` 推論時間の振れ調査

関連 Issue / PR: #446 (フェーズ別計測, #448 でマージ), #447 (GPU バッファ再利用仮説, PR #449 で効果なく close)

> **Note (2026-04-18 更新)**: 本資料は当初 "PyTorch caching allocator のブロック再編成" を真因として pochitrain PR #440 方式での解決を提示していたが, 実機検証と新エビデンスで **仮説が誤りと確認された**. 真因領域は **asyncio / uvicorn / HTTP serving 経路 (特に Windows)** に絞り込まれた. 本改訂は経緯と最新の仮説を整理する.

## 概要

`POST /api/v1/detect` の `pipeline_inference_ms` (TRT execute_async_v3 + stream sync の wall clock) が **38ms / 80ms / 100ms の三段階 bimodal 振動** を示す. 同じ TRT engine を使う CLI / OpenCV カメラストリームでは 8-24ms で安定するため, 問題は HTTP serving 経路に固有と判明した. pochitrain の同名課題 (caching allocator 起因) とは別物であり, 姉妹プロジェクトの既存対策はそのまま適用できない.

## 検証環境

- OS: Windows 11 Home
- GPU: NVIDIA GeForce (ノート PC)
- Python 3.13
- pochidetection dev ブランチ (#448 マージ後)
- モデル: RT-DETR (TensorRT engine, 640×640)
- テストデータ: `data/val/JPEGImages/` の 36 枚 (512×512 〜 一般撮影サイズ)

## 観測の全体像

同一 TRT engine について:

| 経路 | inference 時間 (ms) | 安定性 | gap |
|---|---|---|---|
| Standalone tight loop | 8.0 | 安定 | ~0 |
| CLI (`pochi infer`) | 8.4 | 安定 | ~0 |
| **OpenCV カメラ (PyTorch backend)** | 24.77 平均 (244 frames) | 安定 | ~33ms |
| **OpenCV カメラ (TRT backend)** | **14.17 平均 (182 frames)** | **安定** | ~40ms |
| API HTTP (FastAPI async def) | 38-150 | **bimodal 振動** | 可変 |

ソース: `work_dirs/pretrained/inference_003/stream_metadata.json` (PyTorch), `work_dirs/20260228_001/best/inference_024/stream_metadata.json` (TRT).

**決定的な事実**:
- gap 40ms ありの TRT カメラが 14ms 安定 ⇒ 「gap 中の allocator 再編成」仮説は成立しない.
- 同期 Python ループ (CLI / カメラ) は全て安定. **asyncio runtime を経由する経路だけが振動**.

## 経緯

### 1. 初期現象 (PR #440 マージ後)

| 計測対象 | E2E | 推論本体 |
|---|---|---|
| CLI (`pochi infer`) | 12.3ms | 8.4ms 安定 |
| API (`POST /detect`) | 49-90ms | 38-130ms 振動 |

API は CLI 同等のパス (同一 TRT engine, 同一 `pipeline.run()`) のはずなのに 5-13 倍遅く値が安定しない.

### 2. フェーズ別計測の追加 (#446 → PR #448)

`/detect` の各処理を個別計測する仕組みを追加し `DetectResponse.phase_times_ms` に格納.

```
1. base64.b64decode             (serializers.py)
2. cv2.imdecode (jpeg のみ)      (serializers.py)
3. request.model_dump()         (routers/inference.py)
4. cv2.cvtColor BGR→RGB         (backends.py)
5. Image.fromarray              (rtdetr_pipeline.py)
6. v2.Compose (Resize+ToTensor) (rtdetr_pipeline.py)
7. inference (TRT execute)      (rtdetr_pipeline.py)  ← 振動はここに集中
8. postprocess (HF + NMS)       (rtdetr_pipeline.py)
```

これにより preprocess / postprocess は安定で, 振動は `pipeline.infer()` (TRT execute + stream sync) 内部と判明.

### 3. 仮説 A-D の棄却プロセス (#448 時点)

| 仮説 | 検証 | 結果 |
|---|---|---|
| A. `Image.fromarray` PIL コピー | preprocess 単独計測 | ❌ 8.5ms で全体振れを説明できず |
| B. GPU クロック P-state ダウン | standalone + sleep(30ms)×30 | ❌ 8ms 安定 |
| C. `async def` vs `def` ハンドラ | handler 書き換え | ❌ 両方とも bimodal |
| D. TRT context のスレッド紐付け | thread ID ログ | ❌ 同一スレッドでも振れ |

### 4. 暫定真因「caching allocator」と PR #449 (#447) — **後に棄却**

pochitrain Issue #391 / PR #440 と症状が似ているため, **PyTorch caching allocator のブロック再編成が真因** と推定した. 対策として GPU 入力バッファの事前確保 + in-place 再利用 + warmup 3 回化を PR #449 で実装.

### 5. PR #449 の実機検証で無効と判明

36 枚連続ベンチで:
- `pipeline_preprocess_ms`: 7-11ms で安定 (buffer 再利用は動作している)
- `pipeline_inference_ms`: **38-130ms の bimodal 振動は改修前と同等**

dev (改修なし) での再ベンチも同じ振動を示したため, 改修自体がノーオペだったと結論. PR #449 は close, Issue #447 も close.

### 6. カメラ対比で真因領域を asyncio / uvicorn に絞り込み

同一 TRT engine で OpenCV カメラストリーム (同期 Python ループ, capture gap ~40ms) を計測すると **14.17ms 安定** (182 frames). gap の存在が原因なら API と同様に振れるはずだが安定.

⇒ **振動は HTTP serving 経路 (asyncio event loop + uvicorn + Windows timer) に固有**. 姉妹プロジェクトの caching allocator 仮説は本件には適用できない.

### 7. `run_in_executor` 単独の throwaway 実験 (部分効果)

`engine.predict(...)` を `asyncio.run_in_executor(ThreadPoolExecutor(max_workers=1), ...)` に逃がす最小変更を試した. 結果:

| 区間 | inference (ms) | 評価 |
|---|---|---|
| Req 1-2 | 7.97 / 8.54 | ✅ 8ms 復活 (dev の 49ms から大幅改善) |
| Req 3-50 | 37-130 (bimodal) | ❌ 振動は同じ |
| 末尾 3 件 | 175 → 8.16 → 8.22 → 8.39 | ✅ 突然 8ms に落ちた |

asyncio thread と inference thread の分離は **境界では効いている** が, 中盤の bimodal は解けない. 単独では不十分と判断して破棄.

## Web 調査からの仮説

### W1. Windows の monotonic clock 解像度

既定 15.6ms (HPET 有効時 0.5ms). `asyncio.loop.call_later` は最大 1 clock 解像度分早く発火する仕様のため, **Windows では asyncio スケジューリングに ±15.6ms ジッタが常に含まれる**.
出典: [Python docs: Platform Support (Windows)](https://docs.python.org/3/library/asyncio-platforms.html), [Higher resolution timers on Windows? (discuss.python.org)](https://discuss.python.org/t/higher-resolution-timers-on-windows/16153)

### W2. ThreadPool と GIL 競合

Luis Sena の実測: `ThreadPoolExecutor(max_workers=1) + run_in_executor` は ML ライブラリ内部スレッドと GIL を奪い合って **4x 悪化**. `ProcessPoolExecutor` が唯一安定 (7-9ms 定常).
出典: [How to Optimize FastAPI for ML Model Serving (Luis Sena)](https://luis-sena.medium.com/how-to-optimize-fastapi-for-ml-model-serving-6f75fb9e040d)

### W3. GPU 並行で GIL 競合警告

Jonathan Chang: *"Threading might seem tempting, but it can compete with GPU tasks for GIL and hurt GPU utilization."*
出典: [Maximizing PyTorch Throughput with FastAPI (Jonathan Chang)](https://jonathanc.net/blog/maximizing_pytorch_throughput)

### W4. HN 議論: 真の問題は Web 層の同期推論

Triton / TorchServe / BentoML は **推論を別プロセス (or queue 経由の別 worker) に分離** することでこの問題を避ける. FastAPI 直書きは単機能サービスには使えるが, latency 安定を狙うなら queue ベース設計が基本.
出典: [Breaking up with Flask and FastAPI (Hacker News)](https://news.ycombinator.com/item?id=31769316)

## 現時点の仮説

**asyncio event loop thread が GPU stream sync と干渉する** — Windows timer 15.6ms の粒度で asyncio poll cycle が wake し, `cudaStreamSynchronize` を待つ thread の GIL 復帰が遅延する. これが 38 / 80 / 100ms の bimodal (15-16ms の整数倍近い) と整合する可能性.

**反証材料もあり**: `run_in_executor` で event loop thread から分離しても中盤の振動は消えなかった. 単純な thread 分離では解けない別要因がありそう.

## 次の調査軸 (優先度順)

1. **CUDA Event 化の計測導入** — 現状 `pipeline_inference_ms` は wall-clock. `torch.cuda.Event(enable_timing=True)` で GPU 実時間を計測し wall-clock との差分で「Python 待ち時間」を分離. どこが遅いか確定する最優先手.
2. **Dedicated worker thread + `queue.Queue`** — FastAPI handler を「queue に投入して結果を待つ」だけにし, 推論は独立 daemon thread で無限ループ. asyncio と GPU を完全に切り離す.
3. **ProcessPoolExecutor 案** — Luis Sena 実績. ただし TRT engine のファイル (数百 MB) を per-process ロードするため起動コストが重い. 採用は CUDA Event 結果次第.
4. **Windows timer hack** — `ctypes.windll.winmm.timeBeginPeriod(1)` を app.py の lifespan で呼び, timer 解像度を 1ms に強制. 仮説 W1 の検証.
5. **`sys.setswitchinterval(0.001)`** — GIL 切替を 5ms → 1ms に短縮. 仮説 W2 の検証.

## 棄却された仮説の整理

| 仮説 | 検証方法 | 結果 |
|---|---|---|
| A. `Image.fromarray` PIL コピー | preprocess フェーズ単独計測 | ❌ 8.5ms で全体振れを説明できず |
| B. GPU クロック P-state | standalone + `sleep(0.030)` × 30 | ❌ 8ms 安定維持 |
| C. async def vs sync def (event loop) | ハンドラ書き換え + 再ベンチ | ❌ どちらも bimodal |
| D. TRT context スレッド紐付け | `threading.get_ident()` ログ | ❌ async は同一スレッド |
| **E. PyTorch caching allocator 再編成** (pochitrain 流用) | PR #449 で実装 → 36 枚ベンチ | ❌ preprocess 側は安定したが inference 振動は同等. dev 比較でも差なし |
| F. 「gap 中の allocator 再配置」 | カメラ (gap 40ms) で 14ms 安定を確認 | ❌ gap 自体は振動の原因ではない |

## 教訓と再発防止

### 姉妹プロジェクトの流用は仮説として扱う

pochitrain Issue #391 / PR #440 の解決策が pochidetection にも効くと思い込み, `run_in_executor` や CUDA Event 計測の前に実装に進んでしまった. **pochitrain は PyTorch backend 中心, pochidetection は FastAPI + TRT の組み合わせ** という差を軽視した. 次回は最低でも `torch.cuda.Event` による GPU 実時間計測を事前に入れて, 真因が GPU 側か Python 側かを切り分ける.

### Tight loop ベンチだけに頼らない

スタンドアロンが 8ms で安定したことを「動作の正常性の証明」として扱ったが, **本番経路 (HTTP + asyncio) で再現しない実験は意味が薄い**. ベンチ設計に「gap + 別スレッドから呼ぶ」シナリオを含めるべきだった. 現在はカメラストリームが代替ベンチとして機能している.

### 計測の wall-clock と GPU 時間を常に分離

`time.perf_counter()` は CPU 側の待ち時間を含む. `torch.cuda.Event` と比較するだけで仮説空間が一気に狭まる. 今後 GPU 関連のレイテンシ調査では最初から両者を並記する.

## 関連リンク

- [pochidetection PR #448](https://github.com/kurorosu/pochidetection/pull/448): `/detect` フェーズ別計測追加 (本資料の初版).
- [pochidetection PR #449](https://github.com/kurorosu/pochidetection/pull/449): GPU 入力バッファ再利用案 (close 済み, 訂正コメント付き).
- [pochidetection Issue #446](https://github.com/kurorosu/pochidetection/issues/446): フェーズ別計測の追加 (close 済み).
- [pochidetection Issue #447](https://github.com/kurorosu/pochidetection/issues/447): caching allocator 仮説 (棄却済みで close, 棄却根拠コメント付き).
- [pochitrain Issue #391](https://github.com/kurorosu/pochitrain/issues/391): 姉妹プロジェクトの同名課題 (原因が異なると判明).
- [pochitrain PR #440](https://github.com/kurorosu/pochitrain/pull/440): 姉妹プロジェクトの対策 (pochidetection には不適用).
- [Python docs: Platform Support (Windows)](https://docs.python.org/3/library/asyncio-platforms.html)
- [Higher resolution timers on Windows? (discuss.python.org)](https://discuss.python.org/t/higher-resolution-timers-on-windows/16153)
- [How to Optimize FastAPI for ML Model Serving (Luis Sena)](https://luis-sena.medium.com/how-to-optimize-fastapi-for-ml-model-serving-6f75fb9e040d)
- [Maximizing PyTorch Throughput with FastAPI (Jonathan Chang)](https://jonathanc.net/blog/maximizing_pytorch_throughput)
- [Breaking up with Flask and FastAPI (Hacker News)](https://news.ycombinator.com/item?id=31769316)
