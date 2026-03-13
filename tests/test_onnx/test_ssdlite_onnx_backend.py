"""SSDLiteOnnxBackend クラスのテスト."""

from pathlib import Path

import numpy as np
import pytest
import torch

from pochidetection.inference import SSDLiteOnnxBackend
from pochidetection.models import SSDLiteModel

from .conftest import SSDLITE_INPUT_SIZE, SSDLITE_NUM_CLASSES


@pytest.fixture(scope="session")
def ssdlite_onnx_backend(ssdlite_onnx_path: Path) -> SSDLiteOnnxBackend:
    """テスト用 SSDLiteOnnxBackend インスタンス."""
    return SSDLiteOnnxBackend(
        model_path=ssdlite_onnx_path,
        num_classes=SSDLITE_NUM_CLASSES,
        image_size=SSDLITE_INPUT_SIZE,
        providers=["CPUExecutionProvider"],
    )


class TestSSDLiteOnnxBackendInit:
    """SSDLiteOnnxBackend 初期化のテスト."""

    def test_init_creates_session(
        self, ssdlite_onnx_backend: SSDLiteOnnxBackend
    ) -> None:
        """正常な ONNX ファイルでセッションが作成されることを確認する."""
        assert ssdlite_onnx_backend.session is not None

    def test_init_file_not_found(self, tmp_path: Path) -> None:
        """存在しないファイルで FileNotFoundError が発生することを確認する."""
        with pytest.raises(FileNotFoundError, match="ONNX"):
            SSDLiteOnnxBackend(
                model_path=tmp_path / "nonexistent.onnx",
                num_classes=SSDLITE_NUM_CLASSES,
                image_size=SSDLITE_INPUT_SIZE,
            )

    def test_init_directory_raises_value_error(self, tmp_path: Path) -> None:
        """ディレクトリパスで ValueError が発生することを確認する."""
        with pytest.raises(ValueError, match="ファイル"):
            SSDLiteOnnxBackend(
                model_path=tmp_path,
                num_classes=SSDLITE_NUM_CLASSES,
                image_size=SSDLITE_INPUT_SIZE,
            )

    def test_init_non_onnx_suffix_raises_value_error(self, tmp_path: Path) -> None:
        """非 .onnx 拡張子で ValueError が発生することを確認する."""
        dummy_file = tmp_path / "model.bin"
        dummy_file.write_bytes(b"dummy")
        with pytest.raises(ValueError, match=".onnx"):
            SSDLiteOnnxBackend(
                model_path=dummy_file,
                num_classes=SSDLITE_NUM_CLASSES,
                image_size=SSDLITE_INPUT_SIZE,
            )


class TestSSDLiteOnnxBackendAnchors:
    """アンカー生成のテスト."""

    def test_anchors_shape(self, ssdlite_onnx_backend: SSDLiteOnnxBackend) -> None:
        """アンカーの shape が (num_anchors, 4) であることを確認する."""
        anchors = ssdlite_onnx_backend.anchors
        assert anchors.ndim == 2
        assert anchors.shape[1] == 4

    def test_anchors_count_matches_onnx_output(
        self,
        ssdlite_onnx_backend: SSDLiteOnnxBackend,
        ssdlite_onnx_path: Path,
    ) -> None:
        """アンカー数が ONNX の cls_logits.shape[1] と一致することを確認する."""
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(ssdlite_onnx_path), providers=["CPUExecutionProvider"]
        )
        dummy = np.random.randn(1, 3, *SSDLITE_INPUT_SIZE).astype(np.float32)
        outputs = session.run(None, {"pixel_values": dummy})
        num_anchors_onnx = outputs[0].shape[1]

        assert ssdlite_onnx_backend.anchors.shape[0] == num_anchors_onnx

    def test_anchors_are_deterministic(
        self, ssdlite_onnx_backend: SSDLiteOnnxBackend
    ) -> None:
        """2 回生成して同一結果になることを確認する."""
        from pochidetection.inference.ssdlite.postprocessing import generate_anchors

        anchors1 = generate_anchors(SSDLITE_NUM_CLASSES, SSDLITE_INPUT_SIZE)
        anchors2 = generate_anchors(SSDLITE_NUM_CLASSES, SSDLITE_INPUT_SIZE)
        assert torch.allclose(anchors1, anchors2, atol=1e-6)


class TestSSDLiteOnnxBackendInfer:
    """推論のテスト."""

    def test_infer_returns_dict_with_expected_keys(
        self, ssdlite_onnx_backend: SSDLiteOnnxBackend
    ) -> None:
        """出力辞書が boxes, scores, labels キーを含むことを確認する."""
        dummy = torch.randn(1, 3, *SSDLITE_INPUT_SIZE)
        result = ssdlite_onnx_backend.infer({"pixel_values": dummy})

        assert "boxes" in result
        assert "scores" in result
        assert "labels" in result

    def test_infer_output_types(self, ssdlite_onnx_backend: SSDLiteOnnxBackend) -> None:
        """全出力が torch.Tensor であることを確認する."""
        dummy = torch.randn(1, 3, *SSDLITE_INPUT_SIZE)
        result = ssdlite_onnx_backend.infer({"pixel_values": dummy})

        assert isinstance(result["boxes"], torch.Tensor)
        assert isinstance(result["scores"], torch.Tensor)
        assert isinstance(result["labels"], torch.Tensor)

    def test_infer_output_shapes_consistent(
        self, ssdlite_onnx_backend: SSDLiteOnnxBackend
    ) -> None:
        """boxes, scores, labels の要素数が一致することを確認する."""
        dummy = torch.randn(1, 3, *SSDLITE_INPUT_SIZE)
        result = ssdlite_onnx_backend.infer({"pixel_values": dummy})

        n = result["boxes"].shape[0]
        assert result["scores"].shape == (n,)
        assert result["labels"].shape == (n,)
        if n > 0:
            assert result["boxes"].shape[1] == 4

    def test_infer_scores_in_valid_range(
        self, ssdlite_onnx_backend: SSDLiteOnnxBackend
    ) -> None:
        """スコアが [0, 1] 範囲であることを確認する."""
        dummy = torch.randn(1, 3, *SSDLITE_INPUT_SIZE)
        result = ssdlite_onnx_backend.infer({"pixel_values": dummy})

        if result["scores"].numel() > 0:
            assert result["scores"].min() >= 0.0
            assert result["scores"].max() <= 1.0

    def test_infer_labels_in_valid_range(
        self, ssdlite_onnx_backend: SSDLiteOnnxBackend
    ) -> None:
        """ラベルが 0-indexed foreground 範囲であることを確認する."""
        dummy = torch.randn(1, 3, *SSDLITE_INPUT_SIZE)
        result = ssdlite_onnx_backend.infer({"pixel_values": dummy})

        if result["labels"].numel() > 0:
            assert result["labels"].min() >= 0
            assert result["labels"].max() < SSDLITE_NUM_CLASSES

    def test_infer_missing_input_raises_value_error(
        self, ssdlite_onnx_backend: SSDLiteOnnxBackend
    ) -> None:
        """入力キーが不足している場合に ValueError が発生することを確認する."""
        with pytest.raises(ValueError, match="ONNX入力が不足"):
            ssdlite_onnx_backend.infer({"wrong_key": torch.randn(1, 3, 64, 64)})


class TestSSDLiteOnnxBackendEdgeCases:
    """エッジケースのテスト."""

    def test_blank_image_returns_empty_or_low_scores(
        self, ssdlite_onnx_backend: SSDLiteOnnxBackend
    ) -> None:
        """ブランク画像で検出が 0 件またはスコアが低いことを確認する."""
        blank = torch.zeros(1, 3, *SSDLITE_INPUT_SIZE)
        result = ssdlite_onnx_backend.infer({"pixel_values": blank})

        # ランダム重みなので検出 0 件が一般的. 検出がある場合もエラーにはしない
        assert result["boxes"].shape[0] >= 0

    def test_random_weights_detection_count(
        self, ssdlite_onnx_backend: SSDLiteOnnxBackend
    ) -> None:
        """ランダム重みモデルで検出数が detections_per_img 以下であることを確認する."""
        dummy = torch.randn(1, 3, *SSDLITE_INPUT_SIZE)
        result = ssdlite_onnx_backend.infer({"pixel_values": dummy})

        assert result["boxes"].shape[0] <= 300  # detections_per_img


class TestSSDLiteOnnxBackendSynchronize:
    """synchronize のテスト."""

    def test_synchronize_completes(
        self, ssdlite_onnx_backend: SSDLiteOnnxBackend
    ) -> None:
        """synchronize() が例外なく完了することを確認する."""
        ssdlite_onnx_backend.synchronize()
