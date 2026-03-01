"""RT-DETR 推論スクリプト.

学習済みRT-DETRモデルでフォルダ内の画像を一括推論する.
"""

from pathlib import Path
from typing import Any

from PIL import Image
from transformers import RTDetrImageProcessor

from pochidetection.inference import OnnxBackend, PyTorchBackend
from pochidetection.interfaces.backend import IInferenceBackend
from pochidetection.logging import LoggerManager
from pochidetection.models import RTDetrModel
from pochidetection.scripts.rtdetr.inference import (
    DetectionPipeline,
    InferenceSaver,
    Visualizer,
)
from pochidetection.utils import (
    BenchmarkResult,
    PhasedTimer,
    WorkspaceManager,
    build_benchmark_result,
    write_benchmark_result,
)
from pochidetection.visualization import LabelMapper

logger = LoggerManager().get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def _is_onnx_model(model_path: Path) -> bool:
    """モデルパスが ONNX ファイルかどうかを判定する.

    Args:
        model_path: モデルのパス.

    Returns:
        .onnx ファイルの場合 True.
    """
    return model_path.suffix.lower() == ".onnx"


def _load_processor(model_path: Path, config: dict[str, Any]) -> RTDetrImageProcessor:
    """画像前処理プロセッサを読み込む.

    ONNX モデルの場合, processor ファイルは ONNX ファイルと同じディレクトリから
    読み込みを試み, 見つからなければ config の model_name からフォールバックする.

    Args:
        model_path: モデルのパス.
        config: 設定辞書.

    Returns:
        RTDetrImageProcessor インスタンス.

    Raises:
        RuntimeError: processor が解決できない場合.
    """
    if not _is_onnx_model(model_path):
        return RTDetrImageProcessor.from_pretrained(model_path)

    processor_dir = model_path.parent
    processor_config = processor_dir / "preprocessor_config.json"
    if processor_config.exists():
        logger.info(f"Loading processor from {processor_dir}")
        return RTDetrImageProcessor.from_pretrained(processor_dir)

    model_name = config.get("model_name")
    if model_name:
        logger.info(f"Loading processor from model_name: {model_name}")
        return RTDetrImageProcessor.from_pretrained(model_name)

    raise RuntimeError(
        f"RTDetrImageProcessor を解決できません. "
        f"{processor_dir} に preprocessor_config.json が見つからず, "
        f"config に model_name も指定されていません."
    )


def _create_backend(
    model_path: Path, config: dict[str, Any]
) -> tuple[IInferenceBackend, str, bool]:
    """モデルパスからバックエンドを生成する.

    Args:
        model_path: モデルのパス.
        config: 設定辞書.

    Returns:
        (backend, precision, use_fp16) のタプル.
    """
    device = config["device"]
    use_fp16 = config.get("use_fp16", False)

    if _is_onnx_model(model_path):
        logger.info("ONNX backend selected")
        return OnnxBackend(model_path), "fp32", False

    model = RTDetrModel(str(model_path))
    model.to(device)
    model.eval()

    if use_fp16 and device == "cuda":
        model.half()
        logger.info("FP16 enabled")

    precision = "fp16" if (use_fp16 and device == "cuda") else "fp32"
    return PyTorchBackend(model), precision, use_fp16


def infer(
    config: dict[str, Any],
    image_dir: str,
    threshold: float = 0.5,
    model_dir: str | None = None,
) -> None:
    """フォルダ内の画像を一括推論.

    Args:
        config: 設定辞書.
        image_dir: 推論対象の画像フォルダパス.
        threshold: 検出信頼度閾値.
        model_dir: モデルディレクトリ. Noneの場合は最新ワークスペースのbestを使用.
    """
    device = config["device"]

    model_path = _resolve_model_path(config, model_dir)
    if model_path is None:
        return

    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return

    image_files = [
        f for f in image_dir_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        logger.warning(f"No image files found in {image_dir}")
        return

    logger.info(f"Found {len(image_files)} images in {image_dir}")
    logger.info(f"Loading model from {model_path}")

    if config.get("cudnn_benchmark", False) and device == "cuda":
        import torch

        torch.backends.cudnn.benchmark = True
        logger.info("cudnn.benchmark enabled")

    processor = _load_processor(model_path, config)
    backend, precision, use_fp16 = _create_backend(model_path, config)
    runtime_device = "cpu" if _is_onnx_model(model_path) else device

    phased_timer = PhasedTimer(
        phases=DetectionPipeline.PHASES,
        device=runtime_device,
    )
    pipeline = DetectionPipeline(
        backend=backend,
        processor=processor,
        device=runtime_device,
        threshold=threshold,
        use_fp16=use_fp16,
        phased_timer=phased_timer,
    )

    class_names = config.get("class_names")
    label_mapper = LabelMapper(class_names) if class_names else None
    visualizer = Visualizer(label_mapper=label_mapper)

    saver_base = model_path.parent if _is_onnx_model(model_path) else model_path
    saver = InferenceSaver(saver_base)

    logger.info(f"Results will be saved to {saver.output_dir}")

    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")
        detections = pipeline.run(image)
        result_image = visualizer.draw(image, detections)
        output_path = saver.save(result_image, image_file.name)

        inf_timer = phased_timer.get_timer("inference")
        logger.info(
            f"  {image_file.name} ({inf_timer.last_time_ms:.1f}ms) - "
            f"{len(detections)} objects -> {output_path.name}"
        )

    result = build_benchmark_result(
        phased_timer=phased_timer,
        num_images=len(image_files),
        device=device,
        precision=precision,
        model_path=str(model_path),
    )

    json_path = write_benchmark_result(saver.output_dir, result)
    logger.info(f"Benchmark result saved to {json_path}")

    _log_benchmark_summary(result)
    logger.info(f"Results saved to {saver.output_dir}")


def _log_benchmark_summary(result: BenchmarkResult) -> None:
    """ベンチマーク結果のサマリーをログ出力する.

    Args:
        result: ベンチマーク結果.
    """
    m = result.metrics
    s = result.samples
    logger.info(
        f"Inference completed: {s.num_samples} images "
        f"({s.warmup_samples} warmup skipped), "
        f"avg {m.avg_e2e_ms:.1f}ms/image (E2E), "
        f"throughput {m.throughput_e2e_ips:.1f} IPS (E2E), "
        f"{m.throughput_inference_ips:.1f} IPS (inference)"
    )
    for phase_name, phase in m.phases.items():
        logger.info(
            f"  {phase_name}: avg {phase.average_ms:.1f}ms, "
            f"total {phase.total_ms:.1f}ms ({phase.count} measured)"
        )


def _resolve_model_path(
    config: dict[str, Any],
    model_dir: str | None,
) -> Path | None:
    """モデルパスを解決.

    Args:
        config: 設定辞書.
        model_dir: 指定されたモデルディレクトリ.

    Returns:
        モデルパス. エラー時はNone.
    """
    if model_dir is not None:
        model_path = Path(model_dir)
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return None
        return model_path

    workspace_manager = WorkspaceManager(config["work_dir"])
    workspaces = workspace_manager.get_available_workspaces()

    if not workspaces:
        logger.error("No trained models found. Please run training first.")
        return None

    latest_workspace = Path(str(workspaces[-1]["path"]))
    model_path = latest_workspace / "best"

    if not model_path.exists():
        logger.error(
            f"Best model not found at {model_path}. Please run training first."
        )
        return None

    return model_path
