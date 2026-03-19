"""ONNX Runtime Execution Providers 解決のテスト."""

import onnxruntime as ort

from pochidetection.inference.providers import resolve_providers


class TestResolveProviders:
    """resolve_providers のテスト."""

    def test_cpu_returns_cpu_only(self) -> None:
        """CPU 指定時に CPUExecutionProvider のみ返す."""
        providers = resolve_providers("cpu")
        assert providers == ["CPUExecutionProvider"]

    def test_cuda_always_includes_cpu_fallback(self) -> None:
        """CUDA 指定時に CPUExecutionProvider が含まれる."""
        providers = resolve_providers("cuda")
        assert "CPUExecutionProvider" in providers

    def test_cuda_includes_cuda_if_available(self) -> None:
        """CUDA EP が利用可能なら CUDAExecutionProvider が含まれる."""
        available = ort.get_available_providers()
        providers = resolve_providers("cuda")

        if "CUDAExecutionProvider" in available:
            assert providers[0] == "CUDAExecutionProvider"
            assert providers[1] == "CPUExecutionProvider"
        else:
            assert providers == ["CPUExecutionProvider"]

    def test_cpu_result_is_list(self) -> None:
        """戻り値がリスト型である."""
        providers = resolve_providers("cpu")
        assert isinstance(providers, list)
