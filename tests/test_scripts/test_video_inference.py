"""VideoReader / VideoWriter / process_frames のテスト."""

from pathlib import Path

import cv2
import numpy as np

from pochidetection.scripts.common.video import VideoReader, VideoWriter


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
        import pytest

        with pytest.raises(FileNotFoundError):
            VideoReader(Path("/nonexistent/video.mp4"))


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
