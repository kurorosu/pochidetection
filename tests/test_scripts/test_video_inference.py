"""VideoReader / VideoWriter / StreamReader / DisplaySink / CompositeSink のテスト."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from pochidetection.scripts.common.video import (
    CompositeSink,
    DisplaySink,
    StreamReader,
    VideoReader,
    VideoWriter,
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
