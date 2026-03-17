"""infer コマンドの実行ロジック."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from pochidetection.cli.parser import DEFAULT_CONFIG
from pochidetection.configs.schemas import DetectionConfigDict
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


def _resolve_infer(
    config: DetectionConfigDict,
) -> Callable[[DetectionConfigDict, str, str | None, str | None], None]:
    """Architecture に基づいて infer 関数を返す.

    Args:
        config: 設定辞書.

    Returns:
        infer 関数.
    """
    arch = config.get("architecture", "RTDetr")
    if arch == "SSDLite":
        from pochidetection.scripts.ssdlite.infer import infer as ssdlite_infer

        return ssdlite_infer

    if arch == "SSD300":
        from pochidetection.scripts.ssd300.infer import infer as ssd300_infer

        return ssd300_infer

    from pochidetection.scripts.rtdetr.infer import infer as rtdetr_infer

    return rtdetr_infer


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
    from pochidetection.logging import LoggerManager
    from pochidetection.scripts.common.inference import (
        PRETRAINED,
        resolve_model_path,
    )
    from pochidetection.scripts.common.video import (
        VideoReader,
        VideoWriter,
        process_frames,
    )

    logger = LoggerManager().get_logger(__name__)

    model_path = resolve_model_path(config, model_dir)
    if model_path is None:
        return

    video_file = Path(video_path)
    if not video_file.exists():
        logger.error(f"Video file not found: {video_path}")
        return

    # プリトレイン時は config とパイプラインを差し替え
    if model_path == PRETRAINED:
        from pochidetection.scripts.common.coco_classes import PRETRAINED_CONFIG_PATH
        from pochidetection.scripts.rtdetr.infer import (
            _setup_pipeline as rtdetr_setup_pipeline,
        )

        config = ConfigLoader.load(PRETRAINED_CONFIG_PATH)
        setup_pipeline_fn = rtdetr_setup_pipeline
        logger.info("Loading RT-DETR COCO pretrained model")
    else:
        infer_fn = _resolve_infer(config)
        # 各アーキテクチャの _setup_pipeline を取得
        import importlib

        module = importlib.import_module(infer_fn.__module__)
        setup_pipeline_fn = module._setup_pipeline  # type: ignore[attr-defined]
        logger.info(f"Loading model from {model_path}")

    ctx = setup_pipeline_fn(config, model_path)

    # 出力パス: 入力と同じディレクトリに _result 付きで出力
    output_path = video_file.parent / f"{video_file.stem}_result.mp4"

    reader = VideoReader(video_file)
    writer = VideoWriter(output_path, fps=reader.fps, frame_size=reader.frame_size)

    logger.info(
        f"Video: {video_file.name} "
        f"({reader.frame_size[0]}x{reader.frame_size[1]}, "
        f"{reader.fps:.1f}fps, {reader.total_frames} frames)"
    )
    if interval > 1:
        logger.info(f"Frame interval: {interval} (process every {interval} frames)")

    process_frames(
        source=reader,
        sink=writer,
        pipeline=ctx.pipeline,
        visualizer=ctx.visualizer,
        interval=interval,
        logger=logger,
    )

    reader.release()
    writer.release()
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

    if is_video_file(input_path):
        _run_video_infer(config, input_path, args.model_dir, config_path, args.interval)
    else:
        infer_fn = _resolve_infer(config)
        infer_fn(config, input_path, args.model_dir, config_path)
