"""動画・ストリーム推論の共通ロジック.

VideoReader, VideoWriter, StreamReader, DisplaySink, CompositeSink,
process_frames, FrameProcessingResult を提供する.
"""

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pochidetection.interfaces.frame_sink import IFrameSink
from pochidetection.interfaces.frame_source import IFrameSource
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.scripts.common.visualizer import Visualizer
from pochidetection.utils.resource_monitor import (
    format_resource_lines,
    get_resource_usage,
)


@dataclass(frozen=True, slots=True)
class FrameProcessingResult:
    """フレーム処理のサマリー結果.

    Attributes:
        processed_frames: 推論処理したフレーム数.
        total_frames: 総フレーム数 (スキップ含む).
        elapsed_seconds: 経過時間 (秒).
        avg_fps: 平均 FPS.
        phase_summary: フェーズ別計測サマリー (PhasedTimer 未使用時は None).
    """

    processed_frames: int
    total_frames: int
    elapsed_seconds: float
    avg_fps: float
    phase_summary: dict[str, dict[str, int | float]] | None


class VideoReader(IFrameSource):
    """動画ファイルからフレームを読み取る IFrameSource 実装.

    Attributes:
        _cap: OpenCV の VideoCapture インスタンス.
    """

    def __init__(self, path: Path) -> None:
        """初期化.

        Args:
            path: 動画ファイルのパス.

        Raises:
            FileNotFoundError: ファイルが存在しない場合.
            RuntimeError: 動画ファイルを開けない場合.
        """
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

    @property
    def fps(self) -> float:
        """フレームレートを取得."""
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def frame_size(self) -> tuple[int, int]:
        """フレームサイズを (width, height) で取得."""
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @property
    def total_frames(self) -> int:
        """総フレーム数を取得."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[np.ndarray]:
        """フレームを順次返すイテレータ."""
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    def release(self) -> None:
        """リソースを解放する."""
        self._cap.release()


class StreamReader(IFrameSource):
    """Webcam / RTSP ストリームからフレームを読み取る IFrameSource 実装.

    Attributes:
        _cap: OpenCV の VideoCapture インスタンス.
    """

    _DEFAULT_FPS: float = 30.0

    def __init__(self, source: int | str) -> None:
        """初期化.

        Args:
            source: デバイス ID (int) または RTSP URL (str).

        Raises:
            RuntimeError: ストリームを開けない場合.
        """
        if isinstance(source, int):
            self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open stream: {source}")

    @property
    def fps(self) -> float:
        """フレームレートを取得.

        Webcam / RTSP で CAP_PROP_FPS が不正な値を返す場合は 30.0 にフォールバック.
        """
        raw = self._cap.get(cv2.CAP_PROP_FPS)
        if raw <= 0:
            return self._DEFAULT_FPS
        return float(raw)

    @property
    def frame_size(self) -> tuple[int, int]:
        """フレームサイズを (width, height) で取得."""
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @property
    def cap(self) -> cv2.VideoCapture:
        """内部の VideoCapture インスタンスを取得."""
        return self._cap

    def apply_camera_settings(
        self,
        fps: int | None = None,
        resolution: list[int] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """カメラの FPS・解像度を設定する.

        設定値と実際値を INFO で出力し, 不一致時は WARNING で出力する.

        Args:
            fps: 目標 FPS (None の場合は変更しない).
            resolution: 目標解像度 [width, height] (None の場合は変更しない).
            logger: 警告出力用ロガー.
        """
        if resolution is not None:
            req_w, req_h = resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if logger is not None:
                if (actual_w, actual_h) == (req_w, req_h):
                    logger.info(f"Camera resolution: {actual_w}x{actual_h}")
                else:
                    logger.warning(
                        f"Camera resolution: requested {req_w}x{req_h}, "
                        f"actual {actual_w}x{actual_h}"
                    )

        if fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if logger is not None:
                if actual_fps == fps:
                    logger.info(f"Camera FPS: {actual_fps:.0f}")
                else:
                    logger.warning(
                        f"Camera FPS: requested {fps}, actual {actual_fps:.1f}"
                    )

    def get_camera_properties(self) -> dict[str, float]:
        """カメラプロパティを取得.

        Returns:
            プロパティ名と値の辞書.
        """
        props = {
            "frame_width": cv2.CAP_PROP_FRAME_WIDTH,
            "frame_height": cv2.CAP_PROP_FRAME_HEIGHT,
            "fps": cv2.CAP_PROP_FPS,
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
            "contrast": cv2.CAP_PROP_CONTRAST,
            "saturation": cv2.CAP_PROP_SATURATION,
            "hue": cv2.CAP_PROP_HUE,
            "gain": cv2.CAP_PROP_GAIN,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
            "focus": cv2.CAP_PROP_FOCUS,
            "autofocus": cv2.CAP_PROP_AUTOFOCUS,
            "white_balance": cv2.CAP_PROP_WB_TEMPERATURE,
            "auto_white_balance": cv2.CAP_PROP_AUTO_WB,
        }
        return {name: self._cap.get(prop_id) for name, prop_id in props.items()}

    def __iter__(self) -> Iterator[np.ndarray]:
        """フレームを順次返すイテレータ."""
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    def release(self) -> None:
        """リソースを解放する."""
        self._cap.release()


class DisplaySink(IFrameSink):
    """cv2.imshow によるリアルタイム表示シンク.

    Attributes:
        _window_name: ウィンドウ名.
        _cap: カメラ設定ダイアログ用の VideoCapture 参照.
    """

    def __init__(
        self,
        window_name: str = "pochidetection",
        cap: cv2.VideoCapture | None = None,
    ) -> None:
        """初期化.

        Args:
            window_name: ウィンドウ名.
            cap: カメラ設定ダイアログ表示用の VideoCapture (Windows 限定).
        """
        self._window_name = window_name
        self._cap = cap
        self._overlay_visible = True

    @property
    def overlay_visible(self) -> bool:
        """オーバーレイの表示状態."""
        return self._overlay_visible

    def write(self, frame: np.ndarray) -> None:
        """フレームを表示し, q キーで StopIteration を raise.

        s キーが押された場合, Windows のカメラ設定ダイアログを表示する.
        o キーが押された場合, オーバーレイの表示/非表示をトグルする.

        Args:
            frame: BGR 形式の画像フレーム.

        Raises:
            StopIteration: q キーが押された場合.
        """
        cv2.imshow(self._window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise StopIteration
        if key == ord("s") and self._cap is not None:
            self._cap.set(cv2.CAP_PROP_SETTINGS, 0)
        if key == ord("o"):
            self._overlay_visible = not self._overlay_visible

    def release(self) -> None:
        """ウィンドウを破棄する."""
        cv2.destroyAllWindows()


class CompositeSink(IFrameSink):
    """複数の IFrameSink に同時書き出しする複合シンク.

    --record + DisplaySink を両立する.

    Attributes:
        _sinks: 出力先シンクのリスト.
    """

    def __init__(self, sinks: list[IFrameSink]) -> None:
        """初期化.

        Args:
            sinks: 出力先シンクのリスト.
        """
        self._sinks = sinks

    def write(self, frame: np.ndarray) -> None:
        """全シンクにフレームを書き出す.

        Args:
            frame: BGR 形式の画像フレーム.

        Raises:
            StopIteration: 子シンク (DisplaySink 等) が停止を要求した場合.
        """
        for sink in self._sinks:
            sink.write(frame)

    def release(self) -> None:
        """全シンクを解放する."""
        for sink in self._sinks:
            sink.release()


class VideoWriter(IFrameSink):
    """動画ファイルにフレームを書き出す IFrameSink 実装.

    Attributes:
        _writer: OpenCV の VideoWriter インスタンス.
    """

    def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]) -> None:
        """初期化.

        Args:
            path: 出力動画ファイルのパス.
            fps: フレームレート.
            frame_size: フレームサイズ (width, height).
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)

    def write(self, frame: np.ndarray) -> None:
        """フレームを書き出す.

        Args:
            frame: BGR 形式の画像フレーム.
        """
        self._writer.write(frame)

    def release(self) -> None:
        """リソースを解放する."""
        self._writer.release()


def _find_display_sink(sink: IFrameSink) -> DisplaySink | None:
    """シンクツリーから DisplaySink を探す.

    Args:
        sink: 検索対象のシンク.

    Returns:
        DisplaySink インスタンス. 見つからない場合は None.
    """
    if isinstance(sink, DisplaySink):
        return sink
    if isinstance(sink, CompositeSink):
        for s in sink._sinks:
            if isinstance(s, DisplaySink):
                return s
    return None


def process_frames(
    source: IFrameSource,
    sink: IFrameSink,
    pipeline: IDetectionPipeline,
    visualizer: Visualizer,
    *,
    interval: int = 1,
    overlay_fps: bool = False,
    recording: bool = False,
    logger: logging.Logger,
) -> FrameProcessingResult:
    """フレーム単位で推論・描画・書き出しを行う.

    ソースとシンクは抽象基底クラスで受け取るため,
    動画ファイル・Webcam・RTSP 等を共通ロジックで処理可能.

    Args:
        source: フレーム供給元.
        sink: フレーム出力先.
        pipeline: 推論パイプライン.
        visualizer: 検出結果の描画.
        interval: N フレーム間隔で推論 (1 = 全フレーム処理).
        overlay_fps: True の場合, フレーム左上に実測 FPS を描画.
        recording: True の場合, オーバーレイに "REC" を赤文字で表示.
        logger: ロガー.

    Returns:
        フレーム処理のサマリー結果.
    """
    total = getattr(source, "total_frames", 0)
    processed = 0
    frame_idx = 0
    start_time = time.monotonic()
    display_sink = _find_display_sink(sink)

    # フレーム外フェーズの累積計測用
    capture_total_ms = 0.0
    draw_total_ms = 0.0
    display_total_ms = 0.0
    last_capture_ms = 0.0
    last_draw_ms = 0.0
    last_display_ms = 0.0
    last_e2e_fps = 0.0

    # リソース使用状況 (N フレームごとに更新して負荷を抑える)
    _RESOURCE_UPDATE_INTERVAL = 30
    resource_lines: list[str] = []
    display_end = time.monotonic()

    try:
        for frame in source:
            # capture 計測: 前フレームの display 終了 〜 今フレームの capture 完了
            capture_end = time.monotonic()
            last_capture_ms = (capture_end - display_end) * 1000
            if processed > 0:
                capture_total_ms += last_capture_ms

            frame_start = capture_end

            if interval > 1 and frame_idx % interval != 0:
                sink.write(frame)
                frame_idx += 1
                continue

            # BGR → RGB (パイプライン入力用, PIL 変換なし)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 推論 (pre/infer/post は PhasedTimer が計測)
            detections = pipeline.run(rgb)

            # draw 計測 (OpenCV で元の BGR フレームに直接描画)
            draw_start = time.monotonic()
            result_bgr = frame
            visualizer.draw_cv2(result_bgr, detections)
            last_draw_ms = (time.monotonic() - draw_start) * 1000
            draw_total_ms += last_draw_ms

            # FPS オーバーレイ (前フレームの E2E FPS を表示)
            # o キーでトグル可能 (DisplaySink が存在する場合)
            overlay_visible = (
                display_sink.overlay_visible if display_sink is not None else True
            )
            if overlay_fps and overlay_visible:
                lines = [f"FPS: {last_e2e_fps:.1f}"]
                lines.append(f"capture: {last_capture_ms:.1f}ms")

                phased_timer = pipeline.phased_timer
                if phased_timer is not None:
                    pre_ms = phased_timer.get_timer("preprocess").last_time_ms
                    inf_ms = phased_timer.get_timer("inference").last_time_ms
                    post_ms = phased_timer.get_timer("postprocess").last_time_ms
                    lines.append(f"pre: {pre_ms:.1f}ms")
                    lines.append(f"infer: {inf_ms:.1f}ms")
                    lines.append(f"post: {post_ms:.1f}ms")

                lines.append(f"draw: {last_draw_ms:.1f}ms")
                lines.append(f"display: {last_display_ms:.1f}ms")

                # リソース使用状況 (N フレームごとに更新)
                if processed % _RESOURCE_UPDATE_INTERVAL == 0:
                    resource_lines = format_resource_lines(get_resource_usage())
                lines.extend(resource_lines)

                _draw_overlay_text(result_bgr, lines)

                if recording:
                    _draw_rec_indicator(result_bgr, lines)

            # display 計測
            display_start = time.monotonic()
            sink.write(result_bgr)
            new_display_end = time.monotonic()
            last_display_ms = (new_display_end - display_start) * 1000
            display_total_ms += last_display_ms

            # E2E FPS 更新: 前フレームの display 終了 〜 現フレームの display 終了
            e2e_time = new_display_end - display_end
            last_e2e_fps = 1.0 / e2e_time if e2e_time > 0 else 0.0
            display_end = new_display_end

            processed += 1
            frame_idx += 1

            # 進捗ログ (動画ファイルのみ, 100 フレームごと)
            if total > 0 and processed % 100 == 0:
                pct = frame_idx / total * 100
                logger.info(f"Processing: {frame_idx}/{total} frames ({pct:.1f}%)")
    except (StopIteration, KeyboardInterrupt):
        pass
    finally:
        elapsed = time.monotonic() - start_time
        avg_fps = processed / elapsed if elapsed > 0 else 0.0
        logger.info(
            f"Video inference completed: {processed} frames processed "
            f"({frame_idx} total), {elapsed:.1f}s, {avg_fps:.1f} avg E2E FPS"
        )

        # フェーズ別サマリー
        phase_summary = _build_phase_summary(
            pipeline, processed, capture_total_ms, draw_total_ms, display_total_ms
        )
        if phase_summary is not None:
            for phase_name, stats in phase_summary.items():
                logger.info(
                    f"  {phase_name}: avg {stats['average_ms']:.1f}ms, "
                    f"total {stats['total_ms']:.1f}ms "
                    f"({stats['count']} measured)"
                )

    return FrameProcessingResult(
        processed_frames=processed,
        total_frames=frame_idx,
        elapsed_seconds=elapsed,
        avg_fps=avg_fps,
        phase_summary=phase_summary,
    )


def _draw_overlay_text(
    frame: np.ndarray,
    lines: list[str],
    *,
    x: int = 10,
    y_start: int = 20,
    line_height: int = 20,
    font_scale: float = 0.5,
) -> None:
    """白縁取り + 黒文字でオーバーレイテキストを描画する.

    Args:
        frame: BGR 形式の画像フレーム.
        lines: 描画するテキスト行のリスト.
        x: テキストの x 座標.
        y_start: 最初の行の y 座標.
        line_height: 行間のピクセル数.
        font_scale: フォントスケール.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(lines):
        y = y_start + i * line_height
        # 白アウトライン
        cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), 3)
        # 黒文字
        cv2.putText(frame, line, (x, y), font, font_scale, (0, 0, 0), 1)


def _draw_rec_indicator(
    frame: np.ndarray,
    overlay_lines: list[str],
    *,
    x: int = 10,
    y_start: int = 20,
    line_height: int = 20,
    font_scale: float = 0.5,
) -> None:
    """オーバーレイの最下段に赤文字で "REC" を描画する.

    Args:
        frame: BGR 形式の画像フレーム.
        overlay_lines: 既に描画済みのオーバーレイ行 (y 座標計算用).
        x: テキストの x 座標.
        y_start: 最初の行の y 座標.
        line_height: 行間のピクセル数.
        font_scale: フォントスケール.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = y_start + len(overlay_lines) * line_height
    # 白アウトライン
    cv2.putText(frame, "REC", (x, y), font, font_scale, (255, 255, 255), 3)
    # 赤文字 (BGR)
    cv2.putText(frame, "REC", (x, y), font, font_scale, (0, 0, 255), 1)


def _build_phase_summary(
    pipeline: IDetectionPipeline,
    processed: int,
    capture_total_ms: float,
    draw_total_ms: float,
    display_total_ms: float,
) -> dict[str, dict[str, int | float]] | None:
    """全フェーズのサマリーを構築する.

    Args:
        pipeline: 推論パイプライン.
        processed: 処理済みフレーム数.
        capture_total_ms: キャプチャの合計時間 (ms).
        draw_total_ms: 描画の合計時間 (ms).
        display_total_ms: 表示の合計時間 (ms).

    Returns:
        フェーズ別サマリー辞書. フレーム未処理の場合は None.
    """
    if processed == 0:
        return None

    summary: dict[str, dict[str, int | float]] = {
        "capture": {
            "total_ms": capture_total_ms,
            "count": processed,
            "average_ms": capture_total_ms / processed,
        },
    }

    # PhasedTimer のフェーズ (pre/infer/post)
    phased_timer = pipeline.phased_timer
    if phased_timer is not None:
        summary.update(phased_timer.summary())

    summary["draw"] = {
        "total_ms": draw_total_ms,
        "count": processed,
        "average_ms": draw_total_ms / processed,
    }
    summary["display"] = {
        "total_ms": display_total_ms,
        "count": processed,
        "average_ms": display_total_ms / processed,
    }

    return summary
