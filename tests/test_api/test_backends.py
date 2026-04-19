"""IDetectionBackend の公開メソッド群を classical test で検証."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pochidetection.api.backends import (
    OnnxDetectionBackend,
    PyTorchDetectionBackend,
    TrtDetectionBackend,
    create_detection_backend,
)
from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.core.detection import Detection
from pochidetection.interfaces.pipeline import IDetectionPipeline, ImageInput
from pochidetection.models import RTDetrModel


class _StubPipeline(IDetectionPipeline[Any, Any]):
    """run() の戻り値を固定し, GPU event 計測値も設定可能なスタブ.

    ``run_thresholds`` に呼び出しごとの threshold 引数を記録し, backend から
    pipeline への threshold 伝搬を classical に検証できるようにする.
    ``stub_detections`` を渡すと, ``score_threshold`` に応じたフィルタを
    行わずそのまま返却する (2 段フィルタが backend 側で走らないことを
    検証するため).
    """

    def __init__(
        self,
        *,
        gpu_inference_ms: float | None = None,
        stub_detections: list[Detection] | None = None,
    ) -> None:
        """phased_timer なし, last_inference_gpu_ms を初期化する."""
        super().__init__()
        self._validate_phased_timer(None)
        self._last_inference_gpu_ms = gpu_inference_ms
        self.run_calls: list[np.ndarray] = []
        self.run_thresholds: list[float | None] = []
        self._stub_detections = stub_detections or []

    def run(
        self, image: ImageInput, *, threshold: float | None = None
    ) -> list[Detection]:
        """画像と threshold を記録し, 固定の検出リストを返す."""
        assert isinstance(image, np.ndarray)
        self.run_calls.append(image.copy())
        self.run_thresholds.append(threshold)
        return list(self._stub_detections)


def _make_config(
    *,
    class_names: list[str] | None = None,
    height: int = 32,
    width: int = 32,
) -> DetectionConfigDict:
    """テスト用の最小 DetectionConfigDict を構築する."""
    return DetectionConfigDict(
        architecture="stub",
        num_classes=len(class_names) if class_names is not None else 1,
        class_names=class_names if class_names is not None else ["dummy"],
        image_size={"height": height, "width": width},
    )


def _make_backend(
    pipeline: _StubPipeline,
    *,
    config: DetectionConfigDict | None = None,
    model_path: Path | None = None,
) -> PyTorchDetectionBackend:
    """PyTorchDetectionBackend をスタブ pipeline でラップする."""
    return PyTorchDetectionBackend(
        pipeline=pipeline,
        config=config if config is not None else _make_config(),
        model_path=model_path if model_path is not None else Path("dummy.pt"),
    )


class TestPredictPhaseTimes:
    """predict() が返す phase_times のキー構成を検証."""

    def test_predict_omits_gpu_ms_key_when_pipeline_reports_none(self) -> None:
        """CPU 等で pipeline.last_inference_gpu_ms が None なら phase_times に
        pipeline_inference_gpu_ms キーは含まれない.
        """
        pipeline = _StubPipeline(gpu_inference_ms=None)
        backend = _make_backend(pipeline)
        image = np.zeros((32, 32, 3), dtype=np.uint8)

        _, phase_times = backend.predict(image)

        assert "pipeline_inference_gpu_ms" not in phase_times
        # phase_times は 4 値 (CUDA 時) / 3 値 (CPU 時) のみで,
        # cvt_color_ms / pipeline_total_ms 等の breakdown を含まない.
        assert "cvt_color_ms" not in phase_times
        assert "pipeline_total_ms" not in phase_times

    def test_predict_includes_gpu_ms_when_pipeline_reports_value(self) -> None:
        """pipeline.last_inference_gpu_ms が set されていれば phase_times に転記."""
        pipeline = _StubPipeline(gpu_inference_ms=7.42)
        backend = _make_backend(pipeline)
        image = np.zeros((32, 32, 3), dtype=np.uint8)

        _, phase_times = backend.predict(image)

        assert phase_times["pipeline_inference_gpu_ms"] == 7.42


class TestPredictThresholdForwarding:
    """predict() の score_threshold が pipeline.run にそのまま渡ることを検証."""

    def test_predict_forwards_score_threshold_to_pipeline_run(self) -> None:
        """backend.predict(score_threshold=X) → pipeline.run(threshold=X)."""
        pipeline = _StubPipeline()
        backend = _make_backend(pipeline)
        image = np.zeros((32, 32, 3), dtype=np.uint8)

        backend.predict(image, score_threshold=0.3)
        backend.predict(image, score_threshold=0.9)

        assert pipeline.run_thresholds == [0.3, 0.9]

    def test_predict_no_second_stage_filter_returns_pipeline_output_verbatim(
        self,
    ) -> None:
        """pipeline が返した検出を backend が score でフィルタし直さない.

        2 段フィルタ構造 (#444 の解消前) では backend 側で
        ``det.score < score_threshold`` を再適用していたため, pipeline が
        しきい値未満の検出を返した場合にサイレントに落とされていた.
        本テストは pipeline 出力がそのまま predict の結果に反映されることを保証する.
        """
        stub = [
            Detection(box=[0.0, 0.0, 1.0, 1.0], score=0.2, label=0),
            Detection(box=[2.0, 2.0, 3.0, 3.0], score=0.9, label=0),
        ]
        pipeline = _StubPipeline(stub_detections=stub)
        backend = _make_backend(pipeline)
        image = np.zeros((32, 32, 3), dtype=np.uint8)

        results, _ = backend.predict(image, score_threshold=0.8)

        # pipeline 側で threshold=0.8 を適用するのが本来の経路だが,
        # _StubPipeline は thresholding をしないため pipeline 出力がそのまま返る.
        # これにより backend 側での 2 段フィルタが無いことが classical に確認できる.
        assert [round(r["confidence"], 2) for r in results] == [0.2, 0.9]


class TestGetModelInfo:
    """get_model_info() の返却フィールドを検証."""

    def test_get_model_info_returns_architecture_and_class_names(self) -> None:
        """architecture, num_classes, class_names, input_size, backend を返す."""
        pipeline = _StubPipeline()
        backend = _make_backend(
            pipeline,
            config=_make_config(class_names=["dog", "cat"], height=640, width=480),
            model_path=Path("weights/best.pt"),
        )

        info = backend.get_model_info()

        assert info["architecture"] == "stub"
        assert info["num_classes"] == 2
        assert info["class_names"] == ["dog", "cat"]
        assert info["input_size"] == (640, 480)
        assert info["backend"] == "pytorch"
        assert info["model_path"] == str(Path("weights/best.pt"))


class TestSetClassNames:
    """set_class_names() による label 差し替え動作を検証."""

    def test_set_class_names_updates_labels(self) -> None:
        """set_class_names() 後に get_model_info().class_names が更新される."""
        pipeline = _StubPipeline()
        backend = _make_backend(
            pipeline, config=_make_config(class_names=["foo"], height=32, width=32)
        )

        backend.set_class_names(["dog", "cat"])

        info = backend.get_model_info()
        assert info["class_names"] == ["dog", "cat"]
        # class_names property も同期する.
        assert backend.class_names == ["dog", "cat"]

    def test_set_class_names_does_not_mutate_caller_list(self) -> None:
        """呼出側の list を保持せずコピーするため, 外部変更の影響を受けない."""
        pipeline = _StubPipeline()
        backend = _make_backend(pipeline)
        names = ["dog", "cat"]

        backend.set_class_names(names)
        names.append("extra")

        assert backend.class_names == ["dog", "cat"]


class TestWarmup:
    """warmup() の画像形状・例外なし動作を検証."""

    def test_warmup_runs_without_raising(self) -> None:
        """warmup() がダミー画像で例外なく完了し, 設定サイズの画像を pipeline に渡す."""
        pipeline = _StubPipeline()
        backend = _make_backend(pipeline, config=_make_config(height=64, width=48))

        backend.warmup()

        assert len(pipeline.run_calls) == 1
        dummy = pipeline.run_calls[0]
        assert dummy.shape == (64, 48, 3)
        assert dummy.dtype == np.uint8


class TestClose:
    """close() の冪等性を検証."""

    def test_close_is_idempotent(self) -> None:
        """close() を 2 回呼んでも例外が出ない (_ConcreteBackend の close は no-op)."""
        pipeline = _StubPipeline()
        backend = _make_backend(pipeline)

        backend.close()
        backend.close()


class TestBackendName:
    """backend_name class 属性の値を検証."""

    @pytest.mark.parametrize(
        "backend_cls, model_suffix, expected_name",
        [
            (PyTorchDetectionBackend, "dummy.pt", "pytorch"),
            (OnnxDetectionBackend, "dummy.onnx", "onnx"),
            (TrtDetectionBackend, "dummy.engine", "tensorrt"),
        ],
        ids=["pytorch", "onnx", "tensorrt"],
    )
    def test_backend_name_matches_class(
        self,
        backend_cls: type,
        model_suffix: str,
        expected_name: str,
    ) -> None:
        """各 backend 実装クラスが正しい ``backend_name`` を持つ."""
        pipeline = _StubPipeline()
        backend = backend_cls(
            pipeline=pipeline, config=_make_config(), model_path=Path(model_suffix)
        )
        assert backend.backend_name == expected_name


def _build_rtdetr_e2e_config(class_names: list[str]) -> DetectionConfigDict:
    """`rtdetr_model` fixture を load できる最小 DetectionConfigDict."""
    return DetectionConfigDict(
        architecture="RTDetr",
        model_name="PekingU/rtdetr_r18vd",
        num_classes=len(class_names),
        class_names=class_names,
        image_size={"height": 64, "width": 64},
        device="cpu",
        use_fp16=False,
        cudnn_benchmark=False,
        infer_score_threshold=0.0,
        nms_iou_threshold=0.5,
        pipeline_mode="cpu",
    )


@pytest.mark.slow
class TestPyTorchDetectionBackendE2E:
    """実 RT-DETR モデルを使った `PyTorchDetectionBackend` の E2E 検証.

    `rtdetr_model` fixture (`pretrained=False`, `num_classes=2`) を
    `save_pretrained` 形式で tmp_path に保存し, `create_detection_backend()`
    で PyTorch backend を構築してから warmup / predict / メタ情報を検証する.
    MagicMock を使わず実モデル経路を通す (`.claude/rules/testing.md` の
    classical test 方針).
    """

    @pytest.fixture
    def saved_model_dir(self, rtdetr_model: RTDetrModel, tmp_path: Path) -> Path:
        """fixture モデルを `save_pretrained` 形式で tmp_path に保存し, そのパスを返す."""
        save_dir = tmp_path / "rtdetr_e2e_model"
        rtdetr_model.save(save_dir)
        return save_dir

    def test_warmup_runs_without_raising(self, saved_model_dir: Path) -> None:
        """`create_detection_backend()` で構築した backend の warmup が例外なく完了する."""
        config = _build_rtdetr_e2e_config(["cat", "dog"])

        backend = create_detection_backend(saved_model_dir, config)
        try:
            assert isinstance(backend, PyTorchDetectionBackend)
            backend.warmup()
        finally:
            backend.close()

    def test_predict_returns_schema_compliant_detections(
        self, saved_model_dir: Path
    ) -> None:
        """`predict()` が list[dict] を返し, 各要素が必須フィールドを持つ."""
        config = _build_rtdetr_e2e_config(["cat", "dog"])

        backend = create_detection_backend(saved_model_dir, config)
        try:
            backend.warmup()
            image = np.zeros((64, 64, 3), dtype=np.uint8)

            detections, phase_times = backend.predict(image, score_threshold=0.0)
        finally:
            backend.close()

        assert isinstance(detections, list)
        for det in detections:
            assert set(det.keys()) >= {"class_id", "class_name", "confidence", "bbox"}
            assert isinstance(det["class_id"], int)
            assert isinstance(det["class_name"], str)
            assert 0.0 <= det["confidence"] <= 1.0
            assert len(det["bbox"]) == 4
        # phase_times に pipeline 内訳が含まれる.
        assert "pipeline_preprocess_ms" in phase_times
        assert "pipeline_inference_ms" in phase_times
        assert "pipeline_postprocess_ms" in phase_times

    def test_predict_with_score_threshold_one_returns_empty(
        self, saved_model_dir: Path
    ) -> None:
        """`score_threshold=1.0` で confidence 閾値超過の検出はなく空リスト."""
        config = _build_rtdetr_e2e_config(["cat", "dog"])

        backend = create_detection_backend(saved_model_dir, config)
        try:
            backend.warmup()
            image = np.zeros((64, 64, 3), dtype=np.uint8)

            detections, _ = backend.predict(image, score_threshold=1.0)
        finally:
            backend.close()

        assert detections == []

    def test_set_class_names_reflected_in_predict_output(
        self, saved_model_dir: Path
    ) -> None:
        """`set_class_names()` 後の predict で class_name が新しい名前を反映する."""
        config = _build_rtdetr_e2e_config(["orig_a", "orig_b"])

        backend = create_detection_backend(saved_model_dir, config)
        try:
            backend.warmup()
            backend.set_class_names(["alpha", "beta"])
            image = np.zeros((64, 64, 3), dtype=np.uint8)

            detections, _ = backend.predict(image, score_threshold=0.0)
        finally:
            backend.close()

        # 検出が発生した場合, class_name は新しい名前セットに属する.
        for det in detections:
            assert det["class_name"] in {"alpha", "beta", str(det["class_id"])}

    def test_get_model_info_returns_expected_fields(
        self, saved_model_dir: Path
    ) -> None:
        """`get_model_info()` が architecture / num_classes / input_size / backend を返す."""
        config = _build_rtdetr_e2e_config(["cat", "dog"])

        backend = create_detection_backend(saved_model_dir, config)
        try:
            info = backend.get_model_info()
        finally:
            backend.close()

        assert info["architecture"] == "RTDetr"
        assert info["num_classes"] == 2
        assert info["class_names"] == ["cat", "dog"]
        assert info["input_size"] == (64, 64)
        assert info["backend"] == "pytorch"
        assert info["model_path"] == str(saved_model_dir)
