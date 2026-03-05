"""device ユーティリティのテスト."""

from pochidetection.utils.device import is_fp16_available


class TestIsFp16Available:
    """is_fp16_available のテスト."""

    def test_true_when_fp16_and_cuda(self) -> None:
        """use_fp16=True かつ device="cuda" のとき True."""
        assert is_fp16_available(True, "cuda") is True

    def test_false_when_fp16_and_cpu(self) -> None:
        """use_fp16=True かつ device="cpu" のとき False."""
        assert is_fp16_available(True, "cpu") is False

    def test_false_when_no_fp16_and_cuda(self) -> None:
        """use_fp16=False かつ device="cuda" のとき False."""
        assert is_fp16_available(False, "cuda") is False

    def test_false_when_no_fp16_and_cpu(self) -> None:
        """use_fp16=False かつ device="cpu" のとき False."""
        assert is_fp16_available(False, "cpu") is False
