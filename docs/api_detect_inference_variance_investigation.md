# `POST /api/v1/detect` 推論時間の振れ調査

関連 Issue / PR: #446 (フェーズ別計測, #448 でマージ), #447 (GPU バッファ再利用仮説, PR #449 で効果なく close), #451 (CUDA Event 計測導入)

> **Note (2026-04-18 第 2 改訂)**: 真因が **NVIDIA GPU driver の adaptive clock policy (省電力 P-state)** であることが `nvidia-smi --lock-gpu-clocks=2625` の検証で確定した. asyncio / uvicorn 仮説も棄却済み. 本改訂で確定結論と検証結果を追記する.
>
> **第 1 改訂 (2026-04-18)**: 本資料は当初 "PyTorch caching allocator のブロック再編成" を真因として pochitrain PR #440 方式での解決を提示していたが, 実機検証と新エビデンスで仮説が誤りと確認された. 真因領域を asyncio / uvicorn 仮説に絞り込んだ (この仮説も後続の検証で棄却).

## 概要 (確定結論)

`POST /api/v1/detect` の `pipeline_inference_ms` の振動 (38-150ms) は **NVIDIA GPU driver の adaptive clock policy** が原因. 5 FPS 程度の散発負荷では driver が「持続負荷なし」と判定し GPU clock を ~800 MHz に下げるため, 推論が 35-45ms かかる. 連続負荷 (CLI tight loop / カメラ 60FPS) では clock が高い (2625 MHz 付近) ままなので 8-14ms で安定する.

`nvidia-smi --lock-gpu-clocks=2625` で GPU clock を最大固定したところ, **5 FPS でも `pipeline_inference_gpu_ms` が 8-10ms 安定** することを実機で確認した. PyTorch caching allocator や asyncio / uvicorn は無関係.

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
- 同期 Python ループ (CLI / カメラ) は全て安定. 当初は asyncio runtime 仮説に絞り込んだが, 後続の実測 (CUDA Event + GPU clock 監視) で **真因は GPU clock 状態 (高クロック維持 vs 低クロック)** であることが判明.

カメラ ~14ms と CLI 8ms の差はリクエスト間隔の違いによる GPU clock 平均値の差で説明できる. 詳細は後段「真因確定」を参照.

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

### 4. 暫定真因「caching allocator」と PR #449 — 棄却

pochitrain Issue #391 / PR #440 と症状が似ているため "PyTorch caching allocator のブロック再編成" を真因と推定し, GPU 入力バッファ事前確保 + warmup 強化を PR #449 で実装. 36 枚連続ベンチおよび dev 比較で改善が無く ノーオペと判明したため close.

### 5. カメラ対比で GPU 内部時間が振れていることを確認

同一 TRT engine で OpenCV カメラストリーム (同期 Python ループ, capture gap ~40ms) を計測すると **14.17ms 安定** (182 frames). gap の有無は決定要因でないことが分かり, 真因は HTTP serving 経路に固有と推定. asyncio / uvicorn 仮説に絞り込んで `run_in_executor` 等を試したがいずれも不十分.

## 真因確定 (2026-04-18)

### 検証 1: CUDA Event 計測で Python 待ち時間がほぼゼロと判明

#451 で `pipeline_inference_ms` (wall-clock) と並列に `pipeline_inference_gpu_ms` (CUDA Event ベース) を計測. 5 FPS API ベンチで両者を比較すると **wall ≈ gpu (差は 0.04-0.4ms)** で一致.

⇒ Python 側の待ち時間 (GIL / asyncio scheduler / OS timer) は実質ゼロ. **振動は GPU 側の実行時間そのものが変わっている**. asyncio / Windows timer 仮説 (W1-W4) は棄却.

### 検証 2: nvidia-smi で GPU clock が低速に張り付いていることを確認

5 FPS API ベンチ中の `nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw,temperature.gpu --format=csv -l 1`:

| 状態 | clocks.gr | clocks.mem | power |
|---|---|---|---|
| デスクトップ idle (browser 等) | 2625 MHz | 10251 MHz | 39 W |
| 深い idle | 210 MHz | 405 MHz | 11 W |
| **5 FPS API 推論中** | **765-1035 MHz** | **810 MHz** | **24-27 W** |

推論中なのに max (3120 MHz) の **30-40% にしか上がっていない**. driver が「散発負荷」と判定して boost を見送っている.

### 検証 3: GPU clock を強制最大化したら振動消失

```cmd
nvidia-smi --lock-gpu-clocks=2625
```
(管理者 cmd 必須)

を適用して 5 FPS で 80 件ベンチ:

- ロック前: `pipeline_inference_gpu_ms` は **35-50ms 中心** + たまに 8ms スパイク
- ロック後: **8-10ms 中心** + たまに 32-40ms スパイク (頻度激減)

**完全な逆転**. GPU clock 状態が真因と確定.

### 真因モデル

```
高 FPS / 連続負荷 (CLI tight loop, カメラ 60FPS)
  ↓
driver が「持続負荷」と判定, GPU clock 維持 (2000+ MHz)
  ↓
inference 8-14ms 安定

低 FPS / 散発負荷 (API 5 FPS, 人間操作)
  ↓
driver が「軽負荷」と判定, GPU clock 低下 (~800 MHz)
  ↓
inference 35-45ms に低下. たまに boost 復帰で 8ms スパイク
```

仮説 B の「GPU クロック P-state ダウン」は **standalone での単純な sleep test では再現しなかった** ため #448 時点で棄却した. 実際は「複数リクエストが断続的に来るパターン」で driver の load detection が低クロック判定を出し続けるという固有挙動だった.

## 解決策の選択肢

### 1. 運用回避 (副作用最小)

GPU clock を `nvidia-smi --lock-gpu-clocks=<max>` で強制. 管理者権限と PC 再起動 (or `--reset-gpu-clocks`) で解除. 副作用: 常時最大クロックなので idle 時の消費電力 / 発熱増加.

```cmd
:: ベンチ前
nvidia-smi --lock-gpu-clocks=2625

:: 終了後
nvidia-smi --reset-gpu-clocks
```

### 2. NVIDIA Control Panel (プロセス単位)

`Manage 3D settings → Program Settings → python.exe` を追加し「Power management mode」を「Prefer maximum performance」に. ただし Optimus 機ではクロック完全固定にはならず効果は限定的 (実測で改善せず).

### 3. アプリ内 keepalive (案, 未実装)

app.py の lifespan で background task を起動し, 30ms 間隔で軽量 GPU 操作 (`torch.cuda.synchronize()` 等) を打ち続けて driver に「持続負荷」と認識させる. アプリ完結だが常時 GPU を使うので電力負担あり. **別 Issue 化対象**.

### 4. 諦めて低 FPS API 用に許容する

実時間アプリ (camera streaming) では発生しない. 散発呼び出しの場合は 35-45ms を許容できるか業務判断.

## 採用 / 棄却された仮説の整理

| 仮説 | 検証方法 | 結果 |
|---|---|---|
| A. `Image.fromarray` PIL コピー | preprocess フェーズ単独計測 | ❌ 8.5ms で全体振れを説明できず |
| B. GPU クロック P-state ダウン | standalone + `sleep(0.030)` × 30 | ❌ standalone では再現せず. 散発負荷シナリオで真因として復活 (G) |
| C. async def vs sync def (event loop) | ハンドラ書き換え + 再ベンチ | ❌ どちらも bimodal |
| D. TRT context スレッド紐付け | `threading.get_ident()` ログ | ❌ async は同一スレッド |
| E. PyTorch caching allocator 再編成 (pochitrain 流用) | PR #449 で実装 → 36 枚ベンチ | ❌ preprocess 側は安定したが inference 振動は同等 |
| F. 「gap 中の allocator 再配置」 | カメラ (gap 40ms) で 14ms 安定を確認 | ❌ gap 自体は振動の原因ではない |
| asyncio / Windows timer / GIL 競合 | CUDA Event 計測 (#451) で wall ≈ gpu | ❌ Python 側待ち時間ほぼゼロ |
| **G. NVIDIA driver の adaptive clock policy** | nvidia-smi で clock 監視 + `--lock-gpu-clocks=2625` で固定 | ✅ **真因確定**. ロックで 5 FPS でも 8-10ms 安定 |

## 教訓と再発防止

### 姉妹プロジェクトの流用は仮説として扱う

pochitrain Issue #391 / PR #440 の解決策が pochidetection にも効くと思い込み, `run_in_executor` や CUDA Event 計測の前に実装に進んでしまった. **pochitrain は PyTorch backend 中心, pochidetection は FastAPI + TRT の組み合わせ** という差を軽視した. 次回は最低でも `torch.cuda.Event` による GPU 実時間計測を事前に入れて, 真因が GPU 側か Python 側かを切り分ける.

### Tight loop ベンチだけに頼らない

スタンドアロンが 8ms で安定したことを「動作の正常性の証明」として扱ったが, **本番経路 (HTTP + asyncio) で再現しない実験は意味が薄い**. ベンチ設計に「gap + 別スレッドから呼ぶ」シナリオを含めるべきだった. 現在はカメラストリームが代替ベンチとして機能している.

### 計測の wall-clock と GPU 時間を常に分離

`time.perf_counter()` は CPU 側の待ち時間を含む. `torch.cuda.Event` と比較するだけで仮説空間が一気に狭まる. 今後 GPU 関連のレイテンシ調査では最初から両者を並記する.

### GPU clock の監視を最初の一歩にする

nvidia-smi での GPU clock 監視は数十秒で実施できる. **GPU 推論の振動を見たら最初に `nvidia-smi --query-gpu=clocks.gr,power.draw --format=csv -l 1`** を流すべき. 仮説 B を最初に棄却した際に standalone tight loop だけで判定したのは早計だった. 「実運用パターンでは clock 平均が低い」という機種固有の挙動は実機モニタでしか観測できない.

### 仮説の棄却は「条件付き」と扱う

仮説 B (GPU クロック P-state ダウン) は #448 時点で「standalone で再現せず」と棄却したが, 真因として復活した. 棄却は **「その検証条件下では再現せず」** という条件付きで記録すべきで, 検証シナリオが本番経路と乖離していないか定期的に再点検する.

## 関連リンク

- [pochidetection PR #448](https://github.com/kurorosu/pochidetection/pull/448): `/detect` フェーズ別計測追加 (本資料の初版).
- [pochidetection PR #449](https://github.com/kurorosu/pochidetection/pull/449): GPU 入力バッファ再利用案 (close 済み, 訂正コメント付き).
- [pochidetection Issue #446](https://github.com/kurorosu/pochidetection/issues/446): フェーズ別計測の追加 (close 済み).
- [pochidetection Issue #447](https://github.com/kurorosu/pochidetection/issues/447): caching allocator 仮説 (棄却済みで close).
- [pochidetection Issue #451](https://github.com/kurorosu/pochidetection/issues/451): CUDA Event 計測導入.
- [pochitrain Issue #391](https://github.com/kurorosu/pochitrain/issues/391) / [PR #440](https://github.com/kurorosu/pochitrain/pull/440): 姉妹プロジェクトの同名課題 (原因が異なると判明).

## 更新履歴

- 2026-04-18 ([PR #452](https://github.com/kurorosu/pochidetection/pull/452)): 第 2 改訂. CUDA Event 計測結果を反映し, 真因を NVIDIA GPU driver の adaptive clock policy (低クロック維持) と確定. `nvidia-smi --lock-gpu-clocks` による検証と解決策案を追記.
- 2026-04-18 ([PR #450](https://github.com/kurorosu/pochidetection/pull/450)): 第 1 改訂. PyTorch caching allocator 仮説を棄却し, asyncio / uvicorn 仮説に絞り込む形で全面改訂.
- 2026-04-18 ([PR #448](https://github.com/kurorosu/pochidetection/pull/448)): 初版作成. `/detect` フェーズ別計測の追加経緯と仮説 A-D の検証結果を記録.
