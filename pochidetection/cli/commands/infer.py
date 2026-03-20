"""infer コマンドの実行ロジック."""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from pochidetection.cli.parser import DEFAULT_CONFIG
from pochidetection.cli.registry import (
    SetupPipelineFn,
    resolve_infer,
    resolve_setup_pipeline,
)
from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.interfaces.frame_sink import IFrameSink
from pochidetection.logging import LoggerManager
from pochidetection.scripts.common.coco_classes import PRETRAINED_CONFIG_PATH
from pochidetection.scripts.common.inference import (
    PRETRAINED,
    resolve_model_path,
)
from pochidetection.scripts.common.video import (
    CompositeSink,
    DisplaySink,
    FrameProcessingResult,
    StreamReader,
    VideoReader,
    VideoWriter,
    process_frames,
)
from pochidetection.scripts.rtdetr.infer import _setup_pipeline as rtdetr_setup_pipeline
from pochidetection.utils import ConfigLoader
from pochidetection.utils.config_resolver import resolve_config_path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


def is_video_file(path: str) -> bool:
    """パスが動画ファイルかどうかを判定する.

    Args:
        path: ファイルパス.

    Returns:
        動画ファイルの場合 True.
    """
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_webcam_source(path: str) -> bool:
    """整数値なら Webcam デバイス ID と判定する.

    Args:
        path: 入力パス文字列.

    Returns:
        Webcam デバイス ID の場合 True.
    """
    return path.isdigit()


def is_rtsp_source(path: str) -> bool:
    """rtsp:// または http:// で始まるなら RTSP と判定する.

    Args:
        path: 入力パス文字列.

    Returns:
        RTSP / HTTP ストリームの場合 True.
    """
    return path.startswith(("rtsp://", "http://"))


def _run_stream_infer(
    config: DetectionConfigDict,
    source: int | str,
    model_dir: str | None,
    config_path: str | None,
    interval: int,
    record_path: str | None,
) -> None:
    """Webcam / RTSP ストリームのリアルタイム推論を実行する.

    Args:
        config: 設定辞書.
        source: デバイス ID (int) または RTSP URL (str).
        model_dir: モデルディレクトリ.
        config_path: 設定ファイルのパス.
        interval: N フレーム間隔で推論.
        record_path: 録画出力パス (None の場合は表示のみ).
    """
    logger = LoggerManager().get_logger(__name__)

    if model_dir is not None:
        model_path = resolve_model_path(config, model_dir)
        if model_path is None:
            return
    else:
        model_path = PRETRAINED

    # プリトレイン時は config とパイプラインを差し替え
    if model_path == PRETRAINED:
        config_path = PRETRAINED_CONFIG_PATH
        config = ConfigLoader.load(PRETRAINED_CONFIG_PATH)
        setup_pipeline_fn: SetupPipelineFn = rtdetr_setup_pipeline
        logger.info("Loading RT-DETR COCO pretrained model")
    else:
        setup_pipeline_fn = resolve_setup_pipeline(config)
        logger.info(f"Loading model from {model_path}")

    ctx = setup_pipeline_fn(config, model_path)

    camera_props: dict[str, float] = {}

    with StreamReader(source) as reader:
        display = DisplaySink(cap=reader.cap)

        sink: IFrameSink
        if record_path is not None:
            writer = VideoWriter(
                Path(record_path), fps=reader.fps, frame_size=reader.frame_size
            )
            sink = CompositeSink(sinks=[display, writer])
        else:
            sink = display

        logger.info(
            f"Stream: {source} "
            f"({reader.frame_size[0]}x{reader.frame_size[1]}, {reader.fps:.1f}fps)"
        )
        if record_path is not None:
            logger.info(f"Recording to {record_path}")

        with sink:
            result = process_frames(
                source=reader,
                sink=sink,
                pipeline=ctx.pipeline,
                visualizer=ctx.visualizer,
                interval=interval,
                overlay_fps=True,
                logger=logger,
            )

        # カメラプロパティを取得 (リーダー解放前)
        if isinstance(source, int):
            camera_props = reader.get_camera_properties()

    # メタデータを推論フォルダに保存
    _save_stream_metadata(
        ctx.saver.output_dir,
        config_path,
        camera_props,
        result,
        logger,
    )


def _save_stream_metadata(
    output_dir: Path,
    config_path: str | None,
    camera_props: dict[str, float],
    result: FrameProcessingResult | None,
    logger: logging.Logger,
) -> None:
    """ストリーム推論のメタデータを推論フォルダに保存する.

    Args:
        output_dir: 推論結果フォルダ.
        config_path: 設定ファイルのパス.
        camera_props: カメラプロパティ辞書.
        result: フレーム処理結果 (中断時は None).
        logger: ロガー.
    """
    # config.py をコピー
    if config_path is not None:
        src = Path(config_path)
        if src.exists():
            shutil.copy2(src, output_dir / src.name)

    # メタデータ JSON を保存
    metadata: dict[str, object] = {}
    if camera_props:
        metadata["camera_properties"] = camera_props
    if result is not None:
        summary: dict[str, object] = {
            "processed_frames": result.processed_frames,
            "total_frames": result.total_frames,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
            "avg_e2e_fps": round(result.avg_fps, 2),
        }
        if result.phase_summary is not None:
            summary["phases"] = {
                phase: {
                    "average_ms": round(stats["average_ms"], 2),
                    "total_ms": round(stats["total_ms"], 2),
                    "count": stats["count"],
                }
                for phase, stats in result.phase_summary.items()
            }
        metadata["summary"] = summary

    if metadata:
        meta_path = output_dir / "stream_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Stream metadata saved to {output_dir}")


def _run_video_infer(
    config: DetectionConfigDict,
    video_path: str,
    model_dir: str | None,
    config_path: str | None,
    interval: int,
) -> None:
    """動画ファイルの推論を実行する.

    Args:
        config: 設定辞書.
        video_path: 動画ファイルパス.
        model_dir: モデルディレクトリ.
        config_path: 設定ファイルのパス.
        interval: N フレーム間隔で推論.
    """
    logger = LoggerManager().get_logger(__name__)

    if model_dir is not None:
        model_path = resolve_model_path(config, model_dir)
        if model_path is None:
            return
    else:
        model_path = PRETRAINED

    video_file = Path(video_path)
    if not video_file.exists():
        logger.error(f"Video file not found: {video_path}")
        return

    # プリトレイン時は config とパイプラインを差し替え
    if model_path == PRETRAINED:
        config = ConfigLoader.load(PRETRAINED_CONFIG_PATH)
        setup_pipeline_fn: SetupPipelineFn = rtdetr_setup_pipeline
        logger.info("Loading RT-DETR COCO pretrained model")
    else:
        setup_pipeline_fn = resolve_setup_pipeline(config)
        logger.info(f"Loading model from {model_path}")

    ctx = setup_pipeline_fn(config, model_path)

    # 出力パス: 入力と同じディレクトリに _result 付きで出力
    output_path = video_file.parent / f"{video_file.stem}_result.mp4"

    with VideoReader(video_file) as reader:
        with VideoWriter(
            output_path, fps=reader.fps, frame_size=reader.frame_size
        ) as writer:
            logger.info(
                f"Video: {video_file.name} "
                f"({reader.frame_size[0]}x{reader.frame_size[1]}, "
                f"{reader.fps:.1f}fps, {reader.total_frames} frames)"
            )
            if interval > 1:
                logger.info(
                    f"Frame interval: {interval} (process every {interval} frames)"
                )

            process_frames(
                source=reader,
                sink=writer,
                pipeline=ctx.pipeline,
                visualizer=ctx.visualizer,
                interval=interval,
                logger=logger,
            )

    logger.info(f"Output saved to {output_path}")


def run_infer(args: argparse.Namespace) -> None:
    """Infer コマンドを実行する.

    Args:
        args: パース済みの引数.
    """
    config_path = resolve_config_path(args.config, args.model_dir, DEFAULT_CONFIG)
    config = ConfigLoader.load(config_path)
    input_path = args.dir or config.get("infer_image_dir")
    if input_path is None:
        print(
            "Error: 推論対象の画像ディレクトリまたは動画ファイルが未指定です. "
            "-d オプションまたは config の infer_image_dir を設定してください.",
            file=sys.stderr,
        )
        sys.exit(1)

    if is_webcam_source(input_path):
        _run_stream_infer(
            config,
            int(input_path),
            args.model_dir,
            config_path,
            args.interval,
            args.record,
        )
    elif is_rtsp_source(input_path):
        _run_stream_infer(
            config,
            input_path,
            args.model_dir,
            config_path,
            args.interval,
            args.record,
        )
    elif is_video_file(input_path):
        _run_video_infer(config, input_path, args.model_dir, config_path, args.interval)
    else:
        infer_fn = resolve_infer(config)
        infer_fn(config, input_path, args.model_dir, config_path)
