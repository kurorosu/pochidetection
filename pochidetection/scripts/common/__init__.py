"""学習・推論の共通モジュール (整理中: reporting/ 等へ段階移行中)."""

from pochidetection.scripts.common.video import (
    CompositeSink,
    DisplaySink,
    LazyVideoWriter,
    StreamReader,
    VideoReader,
    VideoWriter,
    process_frames,
)

__all__ = [
    "CompositeSink",
    "DisplaySink",
    "LazyVideoWriter",
    "StreamReader",
    "VideoReader",
    "VideoWriter",
    "process_frames",
]
