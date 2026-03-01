"""TensorRTExporterのテスト."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("tensorrt")

from pochidetection.models import RTDetrModel
from pochidetection.onnx import OnnxExporter
from pochidetection.tensorrt import TensorRTExporter

INPUT_SIZE = (64, 64)


@pytest.fixture(scope="session")
def dummy_onnx_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """エクスポート済みダミーONNXファイルを作成するfixture.

    実行時間短縮のため, rtdetr_model ではなく極小のダミーCNNを使用する.
    """
    tmp_dir = tmp_path_factory.mktemp("trt_onnx")
    output_path = tmp_dir / "tiny_model.onnx"

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.conv(pixel_values)

    model = TinyModel()
    model.eval()

    dummy_input = torch.randn(1, 3, *INPUT_SIZE)
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["output"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        export_params=True,
        dynamo=False,
    )

    return Path(output_path)


class TestTensorRTExporter:
    """TensorRTExporterのテスト."""

    def test_init(self) -> None:
        """初期化が正常に行われることを確認."""
        exporter = TensorRTExporter()
        assert exporter is not None
        assert exporter.trt_logger is not None

    def test_export_creates_valid_file(
        self, dummy_onnx_path: Path, tmp_path: Path
    ) -> None:
        """正常にTensorRTエンジンが書き出されることを確認する."""
        exporter = TensorRTExporter()
        output_path = tmp_path / "model.engine"

        result_path = exporter.export(
            onnx_path=dummy_onnx_path,
            output_path=output_path,
            input_size=INPUT_SIZE,
            min_batch=1,
            opt_batch=1,
            max_batch=2,
        )

        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_export_invalid_onnx_path(self, tmp_path: Path) -> None:
        """存在しないONNXパスを指定した場合FileNotFoundErrorが発生することを確認."""
        exporter = TensorRTExporter()
        with pytest.raises(FileNotFoundError):
            exporter.export(
                onnx_path=tmp_path / "non_existent.onnx",
                output_path=tmp_path / "model.engine",
                input_size=INPUT_SIZE,
            )
