"""VideoReader / VideoWriter / StreamReader / DisplaySink / CompositeSink のテスト."""

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest

from pochidetection.core.detection import Detection
from pochidetection.interfaces.frame_sink import IFrameSink
from pochidetection.interfaces.frame_source import IFrameSource
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.reporting.visualizer import Visualizer
from pochidetection.scripts.common.video import (
    CompositeSink,
    DisplaySink,
    FrameProcessingResult,
    StreamReader,
    VideoReader,
    VideoWriter,
    _build_phase_summary,
    _draw_overlay_text,
    process_frames,
)


def _create_test_video(path: Path, num_frames: int = 10) -> None:
    """テスト用の動画ファイルを生成する."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (64, 48))
    for i in range(num_frames):
        frame = np.full((48, 64, 3), i * 25, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestVideoReader:
    """VideoReader のテスト."""

    def test_is_frame_source(self, tmp_path: Path) -> None:
        """IFrameSource を継承している."""
        from pochidetection.interfaces.frame_source import IFrameSource

        video_path = tmp_path / "test.mp4"
        _create_test_video(video_path)
        reader = VideoReader(video_path)
        assert isinstance(reader, IFrameSource)
        reader.release()

    def test_fps(self, tmp_path: Path) -> None:
        """FPS が正しく取得できる."""
        video_path = tmp_path / "test.mp4"
        _create_test_video(video_path)
        reader = VideoReader(video_path)
        assert reader.fps == 30.0
        reader.release()

    def test_frame_size(self, tmp_path: Path) -> None:
        """フレームサイズが (width, height) で取得できる."""
        video_path = tmp_path / "test.mp4"
        _create_test_video(video_path)
        reader = VideoReader(video_path)
        assert reader.frame_size == (64, 48)
        reader.release()

    def test_total_frames(self, tmp_path: Path) -> None:
        """総フレーム数が正しく取得できる."""
        video_path = tmp_path / "test.mp4"
        _create_test_video(video_path, num_frames=5)
        reader = VideoReader(video_path)
        assert reader.total_frames == 5
        reader.release()

    def test_iterate_frames(self, tmp_path: Path) -> None:
        """フレームをイテレートできる."""
        video_path = tmp_path / "test.mp4"
        _create_test_video(video_path, num_frames=3)
        reader = VideoReader(video_path)
        frames = list(reader)
        assert len(frames) == 3
        assert all(f.shape == (48, 64, 3) for f in frames)
        reader.release()

    def test_file_not_found(self) -> None:
        """存在しないファイルで FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            VideoReader(Path("/nonexistent/video.mp4"))


class TestStreamReader:
    """StreamReader のテスト."""

    def test_is_frame_source(self, tmp_path: Path) -> None:
        """IFrameSource を継承している."""
        from pochidetection.interfaces.frame_source import IFrameSource

        video_path = tmp_path / "test.mp4"
        _create_test_video(video_path)
        reader = StreamReader(str(video_path))
        assert isinstance(reader, IFrameSource)
        reader.release()

    def test_fps_with_video_file(self, tmp_path: Path) -> None:
        """動画ファイルから FPS を取得できる."""
        video_path = tmp_path / "test.mp4"
        _create_test_video(video_path)
        reader = StreamReader(str(video_path))
        assert reader.fps == 30.0
        reader.release()

    def test_fps_fallback_default(self, tmp_path: Path) -> None:
        """CAP_PROP_FPS が 0 以下の場合に 30.0 にフォールバックする."""
        video_path = tmp_path / "test.mp4"
        _create_test_video(video_path)
        reader = StreamReader(str(video_path))

        # FPS を 0 に偽装: _cap を差し替え
        original_cap = reader._cap  # noqa: SLF001
        original_get = original_cap.get

        class FakeCapture:
            """FPS を 0 に偽装するラッパー."""

            def get(self, prop_id: int) -> float:
                if prop_id == cv2.CAP_PROP_FPS:
                    return 0.0
                return float(original_get(prop_id))

        reader._cap = FakeCapture()  # type: ignore[assignment]  # noqa: SLF001
        assert reader.fps == 30.0

        # 元に戻してリリース
        reader._cap = original_cap  # noqa: SLF001
        reader.release()

    def test_frame_size(self, tmp_path: Path) -> None:
        """フレームサイズが取得できる."""
        video_path = tmp_path / "test.mp4"
        _create_test_video(video_path)
        reader = StreamReader(str(video_path))
        assert reader.frame_size == (64, 48)
        reader.release()

    def test_iterate_frames(self, tmp_path: Path) -> None:
        """フレームをイテレートできる."""
        video_path = tmp_path / "test.mp4"
        _create_test_video(video_path, num_frames=3)
        reader = StreamReader(str(video_path))
        frames = list(reader)
        assert len(frames) == 3
        reader.release()

    def test_invalid_source_raises_error(self) -> None:
        """無効なソースで RuntimeError."""
        with pytest.raises(RuntimeError):
            StreamReader("rtsp://invalid.example.com/nonexistent")

    def test_no_total_frames(self, tmp_path: Path) -> None:
        """total_frames 属性を持たない."""
        video_path = tmp_path / "test.mp4"
        _create_test_video(video_path)
        reader = StreamReader(str(video_path))
        assert not hasattr(reader, "total_frames")
        reader.release()


class TestDisplaySink:
    """DisplaySink のテスト."""

    def test_is_frame_sink(self) -> None:
        """IFrameSink を継承している."""
        from pochidetection.interfaces.frame_sink import IFrameSink

        sink = DisplaySink()
        assert isinstance(sink, IFrameSink)

    def test_default_window_name(self) -> None:
        """デフォルトウィンドウ名が pochidetection."""
        sink = DisplaySink()
        assert sink._window_name == "pochidetection"  # noqa: SLF001

    def test_custom_window_name(self) -> None:
        """カスタムウィンドウ名を設定できる."""
        sink = DisplaySink(window_name="custom")
        assert sink._window_name == "custom"  # noqa: SLF001


class TestCompositeSink:
    """CompositeSink のテスト."""

    def test_is_frame_sink(self) -> None:
        """IFrameSink を継承している."""
        from pochidetection.interfaces.frame_sink import IFrameSink

        sink = CompositeSink(sinks=[])
        assert isinstance(sink, IFrameSink)

    def test_write_delegates_to_all_sinks(self, tmp_path: Path) -> None:
        """全シンクにフレームが書き出される."""
        path1 = tmp_path / "out1.mp4"
        path2 = tmp_path / "out2.mp4"
        writer1 = VideoWriter(path1, fps=30.0, frame_size=(64, 48))
        writer2 = VideoWriter(path2, fps=30.0, frame_size=(64, 48))
        composite = CompositeSink(sinks=[writer1, writer2])

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        for _ in range(3):
            composite.write(frame)
        composite.release()

        # 両方のファイルに 3 フレーム書き出されている
        for path in (path1, path2):
            cap = cv2.VideoCapture(str(path))
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
            cap.release()

    def test_release_delegates_to_all_sinks(self, tmp_path: Path) -> None:
        """release() が全シンクに伝播する."""
        path1 = tmp_path / "out1.mp4"
        path2 = tmp_path / "out2.mp4"
        writer1 = VideoWriter(path1, fps=30.0, frame_size=(64, 48))
        writer2 = VideoWriter(path2, fps=30.0, frame_size=(64, 48))
        composite = CompositeSink(sinks=[writer1, writer2])

        composite.release()

        # release 後にファイルが存在する (正常に閉じられた)
        assert path1.exists()
        assert path2.exists()


class TestVideoWriter:
    """VideoWriter のテスト."""

    def test_is_frame_sink(self, tmp_path: Path) -> None:
        """IFrameSink を継承している."""
        from pochidetection.interfaces.frame_sink import IFrameSink

        output_path = tmp_path / "output.mp4"
        writer = VideoWriter(output_path, fps=30.0, frame_size=(64, 48))
        assert isinstance(writer, IFrameSink)
        writer.release()

    def test_write_and_verify(self, tmp_path: Path) -> None:
        """書き出した動画が読み取り可能."""
        output_path = tmp_path / "output.mp4"
        writer = VideoWriter(output_path, fps=30.0, frame_size=(64, 48))
        for _ in range(5):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        assert output_path.exists()
        cap = cv2.VideoCapture(str(output_path))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        cap.release()


class TestVideoReaderWriterRoundTrip:
    """VideoReader → VideoWriter のラウンドトリップテスト."""

    def test_round_trip_preserves_frame_count(self, tmp_path: Path) -> None:
        """入力と出力でフレーム数が一致する."""
        input_path = tmp_path / "input.mp4"
        output_path = tmp_path / "output.mp4"
        _create_test_video(input_path, num_frames=7)

        reader = VideoReader(input_path)
        writer = VideoWriter(output_path, fps=reader.fps, frame_size=reader.frame_size)

        for frame in reader:
            writer.write(frame)

        reader.release()
        writer.release()

        output_reader = VideoReader(output_path)
        assert output_reader.total_frames == 7
        output_reader.release()


# --- process_frames / _build_phase_summary / _draw_overlay_text テスト ---


class _ListSource(IFrameSource):
    """テスト用のフレームソース. リストからフレームを返す."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames

    @property
    def fps(self) -> float:
        return 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        h, w = self._frames[0].shape[:2]
        return (w, h)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self._frames

    def release(self) -> None:
        pass


class _RecordingSink(IFrameSink):
    """テスト用のシンク. 書き出されたフレームを記録する."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        pass


class _DummyPipeline(IDetectionPipeline[np.ndarray, list[Detection]]):
    """テスト用のパイプライン. 固定の検出結果を返す."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._validate_phased_timer(None)
        self._detections = detections or []

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def infer(self, inputs: np.ndarray) -> list[Detection]:
        return self._detections

    def postprocess(self, outputs: list[Detection]) -> list[Detection]:
        return outputs

    def run(self, image: np.ndarray | None = None) -> list[Detection]:
        return self._detections


class TestProcessFrames:
    """process_frames 関数のテスト."""

    def _make_frames(self, n: int = 3) -> list[np.ndarray]:
        """テスト用 BGR フレームを生成."""
        return [np.full((48, 64, 3), i * 50, dtype=np.uint8) for i in range(n)]

    def test_basic_flow(self) -> None:
        """基本フロー: 全フレームが処理される."""
        frames = self._make_frames(5)
        source = _ListSource(frames)
        sink = _RecordingSink()
        pipeline = _DummyPipeline()
        visualizer = Visualizer()
        logger = logging.getLogger("test")

        result = process_frames(source, sink, pipeline, visualizer, logger=logger)

        assert result.processed_frames == 5
        assert result.total_frames == 5
        assert result.elapsed_seconds > 0
        assert len(sink.frames) == 5

    def test_interval_skips_frames(self) -> None:
        """interval=2 でフレームがスキップされる."""
        frames = self._make_frames(6)
        source = _ListSource(frames)
        sink = _RecordingSink()
        pipeline = _DummyPipeline()
        visualizer = Visualizer()
        logger = logging.getLogger("test")

        result = process_frames(
            source, sink, pipeline, visualizer, interval=2, logger=logger
        )

        # 6 フレーム中, index 0, 2, 4 が処理される
        assert result.processed_frames == 3
        assert result.total_frames == 6
        # 全フレームが sink に書き出される (スキップフレームも)
        assert len(sink.frames) == 6

    def test_empty_source(self) -> None:
        """0 フレームのソースで正常に返る."""
        source = _ListSource([])
        sink = _RecordingSink()
        pipeline = _DummyPipeline()
        visualizer = Visualizer()
        logger = logging.getLogger("test")

        result = process_frames(source, sink, pipeline, visualizer, logger=logger)

        assert result.processed_frames == 0
        assert result.total_frames == 0
        assert result.avg_fps == 0.0

    def test_result_has_phase_summary(self) -> None:
        """FrameProcessingResult に phase_summary が含まれる."""
        frames = self._make_frames(3)
        source = _ListSource(frames)
        sink = _RecordingSink()
        pipeline = _DummyPipeline()
        visualizer = Visualizer()
        logger = logging.getLogger("test")

        result = process_frames(source, sink, pipeline, visualizer, logger=logger)

        # PhasedTimer なしでも capture/draw/display は含まれる
        assert result.phase_summary is not None
        assert "capture" in result.phase_summary
        assert "draw" in result.phase_summary
        assert "display" in result.phase_summary

    def test_overlay_fps_does_not_crash(self) -> None:
        """overlay_fps=True でクラッシュしない."""
        frames = self._make_frames(3)
        source = _ListSource(frames)
        sink = _RecordingSink()
        pipeline = _DummyPipeline()
        visualizer = Visualizer()
        logger = logging.getLogger("test")

        result = process_frames(
            source, sink, pipeline, visualizer, overlay_fps=True, logger=logger
        )

        assert result.processed_frames == 3


class TestBuildPhaseSummary:
    """_build_phase_summary 関数のテスト."""

    def test_zero_processed_returns_none(self) -> None:
        """処理フレーム 0 で None を返す."""
        pipeline = _DummyPipeline()
        result = _build_phase_summary(pipeline, 0, 0.0, 0.0, 0.0)
        assert result is None

    def test_summary_keys(self) -> None:
        """capture, draw, display キーが含まれる."""
        pipeline = _DummyPipeline()
        result = _build_phase_summary(pipeline, 10, 100.0, 50.0, 30.0)

        assert result is not None
        assert "capture" in result
        assert "draw" in result
        assert "display" in result

    def test_summary_averages(self) -> None:
        """平均値が正しく計算される."""
        pipeline = _DummyPipeline()
        result = _build_phase_summary(pipeline, 10, 100.0, 50.0, 30.0)

        assert result is not None
        assert result["capture"]["average_ms"] == pytest.approx(10.0)
        assert result["draw"]["average_ms"] == pytest.approx(5.0)
        assert result["display"]["average_ms"] == pytest.approx(3.0)

    def test_summary_counts(self) -> None:
        """count が processed と一致する."""
        pipeline = _DummyPipeline()
        result = _build_phase_summary(pipeline, 5, 50.0, 25.0, 15.0)

        assert result is not None
        assert result["capture"]["count"] == 5
        assert result["draw"]["count"] == 5
        assert result["display"]["count"] == 5


class TestDrawOverlayText:
    """_draw_overlay_text 関数のテスト."""

    def test_draws_text_on_frame(self) -> None:
        """テキストがフレームに描画される."""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        original = frame.copy()

        _draw_overlay_text(frame, ["FPS: 30.0", "infer: 10ms"])

        assert not np.array_equal(frame, original)

    def test_empty_lines_no_change(self) -> None:
        """空のリストで画像が変更されない."""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        original = frame.copy()

        _draw_overlay_text(frame, [])

        np.testing.assert_array_equal(frame, original)


class TestFrameProcessingResult:
    """FrameProcessingResult dataclass のテスト."""

    def test_creation(self) -> None:
        """インスタンス生成と属性アクセス."""
        result = FrameProcessingResult(
            processed_frames=100,
            total_frames=120,
            elapsed_seconds=4.0,
            avg_fps=25.0,
            phase_summary={
                "capture": {"total_ms": 100.0, "count": 100, "average_ms": 1.0}
            },
        )

        assert result.processed_frames == 100
        assert result.total_frames == 120
        assert result.elapsed_seconds == 4.0
        assert result.avg_fps == 25.0
        assert result.phase_summary is not None

    def test_frozen(self) -> None:
        """frozen=True で属性変更不可."""
        result = FrameProcessingResult(
            processed_frames=10,
            total_frames=10,
            elapsed_seconds=1.0,
            avg_fps=10.0,
            phase_summary=None,
        )

        with pytest.raises(AttributeError):
            result.processed_frames = 20  # type: ignore[misc]
