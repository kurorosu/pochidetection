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
from PIL import Image

from pochidetection.interfaces.frame_sink import IFrameSink
from pochidetection.interfaces.frame_source import IFrameSource
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.scripts.common.visualizer import Visualizer


@dataclass(frozen=True, slots=True)
class FrameProcessingResult:
    """フレーム処理のサマリー結果.

    Attributes:
        processed_frames: 推論処理したフレーム数.
        total_frames: 総フレーム数 (スキップ含む).
        elapsed_seconds: 経過時間 (秒).
        avg_fps: 平均 FPS.
    """

    processed_frames: int
    total_frames: int
    elapsed_seconds: float
    avg_fps: float


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

    def write(self, frame: np.ndarray) -> None:
        """フレームを表示し, q キーで StopIteration を raise.

        s キーが押された場合, Windows のカメラ設定ダイアログを表示する.

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


def process_frames(
    source: IFrameSource,
    sink: IFrameSink,
    pipeline: IDetectionPipeline,
    visualizer: Visualizer,
    *,
    interval: int = 1,
    overlay_fps: bool = False,
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
        logger: ロガー.

    Returns:
        フレーム処理のサマリー結果.
    """
    total = getattr(source, "total_frames", 0)
    processed = 0
    frame_idx = 0
    start_time = time.monotonic()

    try:
        for frame in source:
            frame_start = time.monotonic()

            if interval > 1 and frame_idx % interval != 0:
                sink.write(frame)  # スキップフレームはそのまま書き出し
                frame_idx += 1
                continue

            # BGR → RGB → PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            # 推論 + 描画
            detections = pipeline.run(pil_image)
            result_image = visualizer.draw(pil_image, detections, inplace=True)

            # PIL → BGR → 書き出し
            result_bgr = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)

            # FPS オーバーレイ
            if overlay_fps:
                frame_time = time.monotonic() - frame_start
                current_fps = 1.0 / frame_time if frame_time > 0 else 0.0
                cv2.putText(
                    result_bgr,
                    f"FPS: {current_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            sink.write(result_bgr)

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

    return FrameProcessingResult(
        processed_frames=processed,
        total_frames=frame_idx,
        elapsed_seconds=elapsed,
        avg_fps=avg_fps,
    )
