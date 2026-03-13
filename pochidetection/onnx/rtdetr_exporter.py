"""ONNXエクスポート機能を提供するモジュール."""

import logging
import warnings
from pathlib import Path

import torch

from pochidetection.logging import LoggerManager
from pochidetection.models import RTDetrModel
from pochidetection.onnx.validation import verify_onnx_outputs

logger: logging.Logger = LoggerManager().get_logger(__name__)


class RTDetrOnnxExporter:
    """PyTorchモデルをONNX形式にエクスポートするクラス.

    Attributes:
        model: RTDetrModelインスタンス.
        device: 使用デバイス.
    """

    def __init__(
        self,
        model: RTDetrModel | None = None,
        device: torch.device | None = None,
    ) -> None:
        """RTDetrOnnxExporterを初期化.

        Args:
            model: RTDetrModelインスタンス. 後からload_modelで設定も可.
            device: 使用デバイス.
        """
        self.model = model
        self.device = device or torch.device("cpu")

    def load_model(self, model_path: Path) -> None:
        """Huggingface save_pretrained形式のディレクトリからモデルを読み込む.

        Args:
            model_path: モデルディレクトリのパス.
        """
        self.model = RTDetrModel(str(model_path))
        self.model.to(self.device)
        logger.info(f"モデルの読み込み完了: {model_path}")

    def export(
        self,
        output_path: Path,
        input_size: tuple[int, int],
        opset_version: int = 17,
    ) -> Path:
        """モデルをONNX形式でエクスポート.

        Args:
            output_path: 出力ファイルパス.
            input_size: 入力サイズ (height, width).
            opset_version: ONNXオペセットバージョン.

        Returns:
            出力ファイルパス.

        Raises:
            ValueError: モデルが設定されていない場合.
        """
        if self.model is None:
            raise ValueError(
                "モデルが設定されていません. "
                "コンストラクタまたはload_model()でモデルを設定してください."
            )

        self.model.eval()
        self.model.to(self.device)

        dummy_input = torch.randn(
            1, 3, input_size[0], input_size[1], device=self.device
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            torch.onnx.export(
                self.model.model,
                (dummy_input,),
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["pixel_values"],
                output_names=["logits", "pred_boxes"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
                    "logits": {0: "batch_size"},
                    "pred_boxes": {0: "batch_size"},
                },
                dynamo=False,
            )

        logger.info(f"ONNX変換完了: {output_path}")

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.debug(f"ファイルサイズ: {file_size_mb:.2f} MB")

        return output_path

    def verify(
        self,
        onnx_path: Path,
        input_size: tuple[int, int],
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ) -> bool:
        """エクスポートしたONNXモデルを検証.

        PyTorchモデルとONNXモデルの出力を比較し, 許容誤差内で一致するかを確認する.

        Args:
            onnx_path: ONNXモデルのパス.
            input_size: 入力サイズ (height, width).
            rtol: 相対許容誤差.
            atol: 絶対許容誤差.

        Returns:
            検証成功の場合True.

        Raises:
            ValueError: モデルが設定されていない場合.
        """
        if self.model is None:
            raise ValueError(
                "モデルが設定されていません. "
                "コンストラクタまたはload_model()でモデルを設定してください."
            )

        dummy_input = torch.randn(
            1, 3, input_size[0], input_size[1], device=self.device
        )

        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            pytorch_outputs = self.model.model(dummy_input)

        pytorch_logits = pytorch_outputs.logits.cpu().numpy()
        pytorch_boxes = pytorch_outputs.pred_boxes.cpu().numpy()

        return verify_onnx_outputs(
            onnx_path=onnx_path,
            pytorch_outputs=[pytorch_logits, pytorch_boxes],
            dummy_input=dummy_input.cpu().numpy(),
            output_names=["logits", "pred_boxes"],
            rtol=rtol,
            atol=atol,
        )
