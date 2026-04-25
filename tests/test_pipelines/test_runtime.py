"""pipelines/runtime.py のテスト."""

from pathlib import Path

import pytest

from pochidetection.pipelines.runtime import resolve_pipeline_mode


class TestResolvePipelineMode:
    """resolve_pipeline_mode のテスト."""

    @pytest.mark.parametrize(
        ("requested", "suffix", "expected"),
        [
            (None, ".pt", "gpu"),
            (None, ".engine", "gpu"),
            (None, ".onnx", "cpu"),
            ("cpu", ".pt", "cpu"),
            ("gpu", ".pt", "gpu"),
            ("gpu", ".engine", "gpu"),
            ("cpu", ".onnx", "cpu"),
        ],
        ids=[
            "pytorch_default_gpu",
            "tensorrt_default_gpu",
            "onnx_default_cpu",
            "pytorch_explicit_cpu",
            "pytorch_explicit_gpu",
            "tensorrt_explicit_gpu",
            "onnx_explicit_cpu",
        ],
    )
    def test_resolves_mode(
        self,
        tmp_path: Path,
        requested: str | None,
        suffix: str,
        expected: str,
    ) -> None:
        """backend 種別と requested の組合せで解決後の経路名が返ることを確認."""
        model_path = tmp_path / f"model{suffix}"
        assert resolve_pipeline_mode(requested, model_path) == expected  # type: ignore[arg-type]

    def test_onnx_with_explicit_gpu_raises_value_error(self, tmp_path: Path) -> None:
        """ONNX backend + 明示 'gpu' は ValueError で起動拒否しメッセージも一致."""
        model_path = tmp_path / "model.onnx"
        with pytest.raises(ValueError, match="ONNX backend"):
            resolve_pipeline_mode("gpu", model_path)
