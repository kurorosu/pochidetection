"""検出バックエンド抽象と 3 実装 (PyTorch / ONNX / TensorRT)."""

import importlib.metadata
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.interfaces.pipeline import IDetectionPipeline
from pochidetection.logging import LoggerManager
from pochidetection.pipelines.model_path import is_onnx_model, is_tensorrt_model
from pochidetection.pipelines.spec import resolve_and_build_pipeline
from pochidetection.utils.infer_debug import InferDebugConfig, save_infer_debug_image

logger = LoggerManager().get_logger(__name__)


def _safe_version(package: str) -> str | None:
    """指定パッケージのバージョンを返す. 未インストールなら None."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_available_backends() -> list[str]:
    """本環境で利用可能なバックエンド名のリストを返す.

    Returns:
        ``["pytorch", "onnx", "tensorrt"]`` の部分集合. 対応するパッケージが
        import 可能なもののみ含む.
    """
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
    """モデルパスの拡張子からバックエンド種別を推論する.

    Args:
        model_path: モデルファイルまたはディレクトリ.

    Returns:
        ``.engine`` → ``"tensorrt"``, ``.onnx`` → ``"onnx"``, それ以外 (ディレクトリ
        / ``.pth`` 等) → ``"pytorch"``.
    """
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
        pipeline: IDetectionPipeline[Any, Any],
        config: DetectionConfigDict,
        model_path: Path,
        infer_debug: InferDebugConfig | None = None,
    ) -> None:
        """Pipeline と共通メタ情報を保持する.

        Args:
            pipeline: 構築済みの検出 pipeline.
            config: 検証済みの設定辞書.
            model_path: モデルファイルまたはディレクトリ.
            infer_debug: preprocess 後画像のデバッグ保存設定. ``None`` で無効.
                マルチリクエスト並行時の counter は本クラスが ``threading.Lock``
                で排他制御する.
        """
        self._pipeline = pipeline
        self._config = config
        self._model_path = model_path
        self._class_names: list[str] = list(config.get("class_names") or [])
        self._infer_debug = infer_debug
        self._infer_debug_saved = 0
        self._infer_debug_lock = threading.Lock()

    @property
    def pipeline(self) -> IDetectionPipeline[Any, Any]:
        """ラップしている検出 pipeline を返す."""
        return self._pipeline

    @property
    def class_names(self) -> list[str]:
        """現在のクラス名リストを返す."""
        return self._class_names

    def set_class_names(self, class_names: list[str]) -> None:
        """ラベル lookup で使用するクラス名を上書きする.

        Args:
            class_names: 新しいクラス名リスト. ``class_id`` のインデックスで参照される.
        """
        self._class_names = list(class_names)

    def get_model_info(self) -> dict[str, Any]:
        """``/model-info`` エンドポイント用のモデルメタ情報を返す.

        Returns:
            ``architecture`` / ``num_classes`` / ``class_names`` / ``input_size``
            (``(height, width)``) / ``model_path`` / ``backend`` を持つ辞書.
        """
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
        """ダミー画像で 1 回推論し, pipeline を warmup する."""
        image_size = self._config["image_size"]
        height, width = image_size["height"], image_size["width"]
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        self._pipeline.run(dummy)

    def predict(
        self,
        image: np.ndarray,
        *,
        score_threshold: float = 0.5,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Run detection on a BGR uint8 image.

        Args:
            image: 入力画像 (H, W, 3) uint8 BGR (cv2 convention).
            score_threshold: 信頼度の下限しきい値. ``pipeline.run`` の
                ``threshold`` 引数にそのまま渡され, config の
                ``infer_score_threshold`` を request 単位で上書きする.

        Returns:
            (検出結果のリスト, フェーズ別所要時間 ms). 検出結果は
            class_id / class_name / confidence / bbox を持つ辞書.
            フェーズ別所要時間は pipeline の PhasedTimer から
            pipeline_preprocess_ms / pipeline_inference_ms /
            pipeline_postprocess_ms を含む. CUDA 利用時は
            ``pipeline_inference_gpu_ms`` (CUDA Event 計測の GPU 実時間) も
            追加される. wall-clock との差分が Python 側の待ち時間
            (GIL / asyncio / OS scheduler) の指標となる.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._maybe_save_infer_debug(rgb)
        raw_detections = self._pipeline.run(rgb, threshold=score_threshold)

        phase_times: dict[str, float] = {}
        pt = self._pipeline.phased_timer
        if pt is not None:
            phase_times["pipeline_preprocess_ms"] = pt.get_timer(
                "preprocess"
            ).last_time_ms
            phase_times["pipeline_inference_ms"] = pt.get_timer(
                "inference"
            ).last_time_ms
            phase_times["pipeline_postprocess_ms"] = pt.get_timer(
                "postprocess"
            ).last_time_ms
        gpu_ms = self._pipeline.last_inference_gpu_ms
        if gpu_ms is not None:
            phase_times["pipeline_inference_gpu_ms"] = gpu_ms

        results: list[dict[str, Any]] = []
        for det in raw_detections:
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
        return results, phase_times

    def _maybe_save_infer_debug(self, rgb: np.ndarray) -> None:
        """推論 request 毎に先頭 N 枚の preprocess 後画像を保存する.

        Lock 下で counter を進め, 上限到達後は何もしない. 保存対象と判定された
        index は lock 外で JPEG 化するため, I/O がロック区間を延ばさない.

        Args:
            rgb: RGB uint8 画像.
        """
        if self._infer_debug is None:
            return

        with self._infer_debug_lock:
            if self._infer_debug_saved >= self._infer_debug.save_count:
                return
            idx = self._infer_debug_saved
            self._infer_debug_saved += 1

        save_infer_debug_image(
            source_image=Image.fromarray(rgb),
            target_hw=self._infer_debug.target_hw,
            letterbox=self._infer_debug.letterbox,
            save_path=self._infer_debug.output_dir / f"infer_{idx:04d}.jpg",
        )

    @abstractmethod
    def close(self) -> None:
        """バックエンドが保持するリソースを解放する.

        明示的に解放する必要のあるリソース (TensorRT engine / context,
        CUDA provider を抱える ONNX session など) を持つ実装のみ, 本メソッドで
        解放処理を行う. PyTorch バックエンドのように GC に任せられる実装は
        no-op で良い.
        """


class _ConcreteBackend(IDetectionBackend):
    """共通の concrete バックエンド.

    構築済みの pipeline をラップするだけの backend 基底. リソース所有権は
    pipeline 側にあるため ``close()`` は no-op で良い.
    """

    def close(self) -> None:
        """No-op. 配下の pipeline がリソース所有権を持つため解放不要."""


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
    """model_path から pipeline を解決し, detection backend を構築する.

    Args:
        model_path: モデルファイルまたはディレクトリ.
        config: ロード済みの設定辞書.
        config_path: 設定ファイルパス. 指定時は推論結果ディレクトリへコピーされる.

    Returns:
        構築済みのバックエンド.

    Raises:
        RuntimeError: パイプライン構築に失敗した場合.
    """
    backend_name = detect_backend_from_model(model_path)
    backend_cls = _BACKEND_CLASSES[backend_name]

    logger.info(f"Loading model: {model_path} (backend={backend_name})")
    resolved = resolve_and_build_pipeline(
        config=config,
        model_dir=str(model_path),
        config_path=config_path,
    )
    if resolved is None:
        raise RuntimeError(f"モデルロードに失敗しました: {model_path}")

    infer_debug = InferDebugConfig.from_config(
        resolved.config, resolved.context.saver.output_dir
    )

    return backend_cls(
        pipeline=resolved.context.pipeline,
        config=resolved.config,
        model_path=resolved.model_path,
        infer_debug=infer_debug,
    )
