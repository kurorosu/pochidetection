"""export コマンドの実行ロジック."""

import argparse
import sys
from pathlib import Path

from pochidetection.cli.parser import DEFAULT_CONFIG
from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.utils import ConfigLoader
from pochidetection.utils.config_resolver import resolve_config_path


def _run_trt_export(args: argparse.Namespace, config: DetectionConfigDict) -> None:
    """Tensorrt エクスポート.

    Args:
        args: パース済みの引数.
        config: 設定辞書.
    """
    model_path = Path(args.model_path)
    input_size_trt: tuple[int, int] = (
        (args.input_size[0], args.input_size[1])
        if args.input_size
        else (
            int(config["image_size"]["height"]),
            int(config["image_size"]["width"]),
        )
    )

    int8_calibrator = None
    if args.int8:
        from pochidetection.tensorrt import INT8Calibrator

        calib_image_dir = Path(str(config["infer_image_dir"]))
        cache_path = model_path.parent / "calibration_cache.bin"
        int8_calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=input_size_trt,
            max_images=args.calib_max_images,
            cache_path=cache_path,
        )

    from pochidetection.scripts.common.export_trt import export_trt

    export_trt(
        args.model_path,
        args.output,
        input_size_trt,
        args.min_batch,
        args.opt_batch,
        args.max_batch,
        args.fp16,
        args.int8,
        int8_calibrator,
        args.build_memory,
    )


def _run_onnx_export(args: argparse.Namespace, config: DetectionConfigDict) -> None:
    """ONNX エクスポートを実行する.

    Args:
        args: パース済みの引数.
        config: 設定辞書.
    """
    input_size = tuple(args.input_size) if args.input_size else None

    if config.get("architecture") == "SSDLite":
        from pochidetection.scripts.ssdlite.export_onnx import (
            export_onnx as ssdlite_export_onnx,
        )

        ssdlite_export_onnx(
            config,
            args.model_path,
            args.output,
            args.opset_version,
            input_size,
            args.skip_verify,
            args.fp16,
        )
    else:
        if args.fp16:
            print(
                "Error: --fp16 は SSDLite の ONNX export でのみ使用できます.",
                file=sys.stderr,
            )
            sys.exit(1)

        from pochidetection.scripts.rtdetr.export_onnx import export_onnx

        export_onnx(
            config,
            args.model_path,
            args.output,
            args.opset_version,
            input_size,
            args.skip_verify,
        )


def run_export(args: argparse.Namespace) -> None:
    """Export コマンドを実行する.

    Args:
        args: パース済みの引数.
    """
    config_path = resolve_config_path(args.config, args.model_path, DEFAULT_CONFIG)
    config = ConfigLoader.load(config_path)

    model_path = Path(args.model_path)
    if model_path.suffix.lower() == ".onnx":
        _run_trt_export(args, config)
    else:
        _run_onnx_export(args, config)
