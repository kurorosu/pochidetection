"""動画推論の共通ロジック.

VideoReader (IFrameSource), VideoWriter (IFrameSink), process_frames を提供する.
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
        logger: ロガー.
    """
    total = getattr(source, "total_frames", 0)
    processed = 0
    frame_idx = 0
    start_time = time.monotonic()

    for frame in source:
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
        sink.write(result_bgr)

        processed += 1
        frame_idx += 1

        # 進捗ログ (100 フレームごと, または total が既知の場合)
        if processed % 100 == 0:
            if total > 0:
                pct = frame_idx / total * 100
                logger.info(f"Processing: {frame_idx}/{total} frames ({pct:.1f}%)")
            else:
                logger.info(f"Processing: {frame_idx} frames")

    elapsed = time.monotonic() - start_time
    avg_fps = processed / elapsed if elapsed > 0 else 0.0
    logger.info(
        f"Video inference completed: {processed} frames processed "
        f"({frame_idx} total), {elapsed:.1f}s, {avg_fps:.1f} avg FPS"
    )
