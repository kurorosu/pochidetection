"""test_tensorrt パッケージ共通フィクスチャ."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("tensorrt")

from pochidetection.tensorrt import RTDetrTensorRTExporter

INPUT_SIZE = (64, 64)


class TinyRTDetrLikeModel(torch.nn.Module):
    """テスト用の極小モデル (RT-DETR と同様に2出力).

    TensorRTBackend.infer() は (pred_logits, pred_boxes) の
    2出力を期待するため, 2つの出力を返すモデルを使用する.
    """

    def __init__(self, num_queries: int = 10, num_classes: int = 3) -> None:
        """初期化."""
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.logits_head = torch.nn.Linear(3, num_queries * num_classes)
        self.boxes_head = torch.nn.Linear(3, num_queries * 4)
        self.num_queries = num_queries
        self.num_classes = num_classes

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """順伝播."""
        batch_size = pixel_values.shape[0]
        x = self.pool(pixel_values).view(batch_size, -1)
        logits = self.logits_head(x).view(
            batch_size, self.num_queries, self.num_classes
        )
        boxes = self.boxes_head(x).view(batch_size, self.num_queries, 4)
        return logits, boxes


@pytest.fixture(scope="session")
def dummy_onnx_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """エクスポート済みダミー ONNX ファイルを作成する fixture.

    実行時間短縮のため, 極小のダミー CNN を使用する.
    """
    tmp_dir = tmp_path_factory.mktemp("trt_onnx")
    output_path = tmp_dir / "tiny_model.onnx"

    model = TinyRTDetrLikeModel()
    model.eval()

    dummy_input = torch.randn(1, 3, *INPUT_SIZE)
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["pred_logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "pred_logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"},
        },
        opset_version=17,
        export_params=True,
        dynamo=False,
    )

    return Path(output_path)


@pytest.fixture(scope="session")
def engine_path(
    dummy_onnx_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """ダミー TensorRT エンジンファイルを作成する fixture."""
    tmp_dir = tmp_path_factory.mktemp("trt_engine")
    output_path = tmp_dir / "tiny_model.engine"

    exporter = RTDetrTensorRTExporter()
    result: Path = exporter.export(
        onnx_path=dummy_onnx_path,
        output_path=output_path,
        input_size=INPUT_SIZE,
        min_batch=1,
        opt_batch=1,
        max_batch=2,
    )

    return result
