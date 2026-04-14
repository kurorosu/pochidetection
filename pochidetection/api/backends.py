"""検出バックエンド抽象と 3 実装 (PyTorch / ONNX / TensorRT)."""

import importlib.metadata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.logging import LoggerManager
from pochidetection.scripts.common.inference import (
    is_onnx_model,
    is_tensorrt_model,
    resolve_and_setup_pipeline,
)

logger = LoggerManager().get_logger(__name__)


def _safe_version(package: str) -> str | None:
    """Return installed package version or None if missing."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_available_backends() -> list[str]:
    """Return backend names available in this environment."""
    available: list[str] = []
    if _safe_version("torch") is not None:
        available.append("pytorch")
    if (
        _safe_version("onnxruntime-gpu") is not None
        or _safe_version("onnxruntime") is not None
    ):
        available.append("onnx")
    if _safe_version("tensorrt") is not None:
        available.append("tensorrt")
    return available


def detect_backend_from_model(
    model_path: Path,
) -> Literal["pytorch", "onnx", "tensorrt"]:
    """Infer backend name from the model path extension."""
    if is_tensorrt_model(model_path):
        return "tensorrt"
    if is_onnx_model(model_path):
        return "onnx"
    return "pytorch"


class IDetectionBackend(ABC):
    """検出バックエンドの抽象基底クラス.

    pochitrain の ``IInferenceBackend`` を検出タスク向けに拡張.
    ``predict()`` は ``list[dict]`` 形式の検出結果を返す.
    """

    backend_name: str = "unknown"

    def __init__(
        self,
        pipeline: IDetectionPipeline,  # type: ignore[type-arg]
        config: DetectionConfigDict,
        model_path: Path,
    ) -> None:
        """Store the pipeline and metadata shared by all backends."""
        self._pipeline = pipeline
        self._config = config
        self._model_path = model_path
        self._class_names: list[str] = list(config.get("class_names") or [])

    @property
    def pipeline(self) -> IDetectionPipeline:  # type: ignore[type-arg]
        """Return the wrapped detection pipeline."""
        return self._pipeline

    @property
    def class_names(self) -> list[str]:
        """Return the current class names."""
        return self._class_names

    def set_class_names(self, class_names: list[str]) -> None:
        """Override class names used for label lookup."""
        self._class_names = list(class_names)

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for the /model-info endpoint."""
        image_size = self._config["image_size"]
        return {
            "architecture": self._config["architecture"],
            "num_classes": self._config["num_classes"],
            "class_names": self._class_names,
            "input_size": (image_size["height"], image_size["width"]),
            "model_path": str(self._model_path),
            "backend": self.backend_name,
        }

    def warmup(self) -> None:
        """Run a single dummy inference to warm up the pipeline."""
        image_size = self._config["image_size"]
        height, width = image_size["height"], image_size["width"]
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        self._pipeline.run(dummy)

    def predict(
        self,
        image: np.ndarray,
        *,
        score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Run detection on a BGR uint8 image.

        Args:
            image: 入力画像 (H, W, 3) uint8 BGR (cv2 convention).
            score_threshold: 信頼度の下限しきい値.

        Returns:
            検出結果のリスト. 各要素は class_id / class_name / confidence / bbox を持つ.
        """
        # IDetectionPipeline は RGB を要求するため BGR から変換する.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_detections = self._pipeline.run(rgb)

        results: list[dict[str, Any]] = []
        for det in raw_detections:
            if det.score < score_threshold:
                continue
            class_id = int(det.label)
            class_name = (
                self._class_names[class_id]
                if 0 <= class_id < len(self._class_names)
                else str(class_id)
            )
            results.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(det.score),
                    "bbox": [float(v) for v in det.box],
                }
            )
        return results

    @abstractmethod
    def close(self) -> None:
        """Release backend-specific resources."""


class _ConcreteBackend(IDetectionBackend):
    """Shared concrete backend that simply wraps a prebuilt pipeline."""

    def close(self) -> None:
        """No-op: underlying pipeline owns its resources."""


class PyTorchDetectionBackend(_ConcreteBackend):
    """PyTorch バックエンド."""

    backend_name = "pytorch"


class OnnxDetectionBackend(_ConcreteBackend):
    """ONNX Runtime バックエンド."""

    backend_name = "onnx"


class TrtDetectionBackend(_ConcreteBackend):
    """TensorRT バックエンド."""

    backend_name = "tensorrt"


_BACKEND_CLASSES: dict[str, type[_ConcreteBackend]] = {
    "pytorch": PyTorchDetectionBackend,
    "onnx": OnnxDetectionBackend,
    "tensorrt": TrtDetectionBackend,
}


def create_detection_backend(
    model_path: Path,
    config: DetectionConfigDict,
    config_path: str | None = None,
) -> IDetectionBackend:
    """Build a detection backend by resolving the pipeline from the model path.

    Args:
        model_path: モデルファイルまたはディレクトリ.
        config: ロード済みの設定辞書.
        config_path: 設定ファイルパス (推論結果ディレクトリへのコピー用, 未使用可).

    Returns:
        構築済みのバックエンド.

    Raises:
        RuntimeError: パイプライン構築に失敗した場合.
    """
    backend_name = detect_backend_from_model(model_path)
    backend_cls = _BACKEND_CLASSES[backend_name]

    logger.info(f"Loading model: {model_path} (backend={backend_name})")
    resolved = resolve_and_setup_pipeline(
        config=config,
        model_dir=str(model_path),
        config_path=config_path,
    )
    if resolved is None:
        raise RuntimeError(f"モデルロードに失敗しました: {model_path}")

    return backend_cls(
        pipeline=resolved.ctx.pipeline,
        config=resolved.config,
        model_path=resolved.model_path,
    )
