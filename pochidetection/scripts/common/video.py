"""動画・ストリーム推論の共通ロジック.

VideoReader, VideoWriter, StreamReader, DisplaySink, CompositeSink,
process_frames を提供する.
"""

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pochidetection.interfaces.frame_sink import IFrameSink
from pochidetection.interfaces.frame_source import IFrameSource
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.scripts.common.visualizer import Visualizer


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
    """

    def __init__(self, window_name: str = "pochidetection") -> None:
        """初期化.

        Args:
            window_name: ウィンドウ名.
        """
        self._window_name = window_name

    def write(self, frame: np.ndarray) -> None:
        """フレームを表示し, q キーで StopIteration を raise.

        Args:
            frame: BGR 形式の画像フレーム.

        Raises:
            StopIteration: q キーが押された場合.
        """
        cv2.imshow(self._window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise StopIteration

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
) -> None:
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
    finally:
        elapsed = time.monotonic() - start_time
        avg_fps = processed / elapsed if elapsed > 0 else 0.0
        logger.info(
            f"Video inference completed: {processed} frames processed "
            f"({frame_idx} total), {elapsed:.1f}s, {avg_fps:.1f} avg FPS"
        )
