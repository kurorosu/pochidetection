"""SSDLite ONNX エクスポート機能を提供するモジュール."""

import copy
import logging
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from pochidetection.logging import LoggerManager
from pochidetection.models import SSDLiteModel
from pochidetection.onnx.validation import verify_onnx_outputs

logger: logging.Logger = LoggerManager().get_logger(__name__)


class _SSDLiteExportWrapper(nn.Module):
    """SSDLite の backbone + head のみをラップする ONNX エクスポート用モジュール.

    torchvision SSD の forward は NMS を含む後処理を行うため,
    ONNX エクスポートには適さない. このラッパーは backbone と head のみを
    呼び出し, NMS 前の生出力 (cls_logits, bbox_regression) を返す.
    """

    def __init__(self, ssd_model: nn.Module) -> None:
        """初期化.

        Args:
            ssd_model: torchvision の SSD モデルインスタンス.
        """
        super().__init__()
        self.backbone = ssd_model.backbone
        self.head = ssd_model.head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Backbone + head の forward.

        GeneralizedRCNNTransform 相当の正規化を適用してから推論する.
        入力は [0, 1] 範囲を期待し, [-1, 1] に変換する.

        Args:
            x: 入力画像テンソル (B, 3, H, W). [0, 1] 範囲.

        Returns:
            (cls_logits, bbox_regression) のタプル.
            cls_logits: (B, num_anchors, num_classes+1).
            bbox_regression: (B, num_anchors, 4).
        """
        x = (x - 0.5) / 0.5
        features = self.backbone(x)
        features_list = list(features.values())
        head_out = self.head(features_list)
        return head_out["cls_logits"], head_out["bbox_regression"]


class SSDLiteOnnxExporter:
    """SSDLite モデルを ONNX 形式にエクスポートするクラス.

    Attributes:
        model: SSDLiteModel インスタンス.
        device: 使用デバイス.
    """

    def __init__(
        self,
        model: SSDLiteModel | None = None,
        device: torch.device | None = None,
    ) -> None:
        """初期化.

        Args:
            model: SSDLiteModel インスタンス. 後から load_model で設定も可.
            device: 使用デバイス.
        """
        self.model = model
        self.device = device or torch.device("cpu")

    def _prepare_wrapper_and_input(
        self,
        input_size: tuple[int, int],
        fp16: bool,
    ) -> tuple[_SSDLiteExportWrapper, torch.Tensor]:
        """Deepcopy からラッパーとダミー入力を生成.

        元モデルへの副作用 (half 変換など) を防止するため,
        モデルを deepcopy してからラッパーを構築する.

        Args:
            input_size: 入力サイズ (height, width).
            fp16: FP16 モードか.

        Returns:
            (wrapper, dummy_input) のタプル.

        Raises:
            ValueError: モデルが設定されていない場合.
        """
        if self.model is None:
            raise ValueError(
                "モデルが設定されていません. "
                "コンストラクタまたは load_model() でモデルを設定してください."
            )

        ssd_copy = copy.deepcopy(self.model.model)
        wrapper = _SSDLiteExportWrapper(ssd_copy)
        wrapper.to(self.device)
        wrapper.eval()

        dtype = torch.float16 if fp16 else torch.float32
        if fp16:
            wrapper.half()

        dummy_input = torch.rand(
            1,
            3,
            input_size[0],
            input_size[1],
            device=self.device,
            dtype=dtype,
        )
        return wrapper, dummy_input

    def load_model(
        self,
        model_path: Path,
        num_classes: int,
        pretrained: bool = True,
        nms_iou_threshold: float = 0.5,
    ) -> None:
        """state_dict 形式のディレクトリからモデルを読み込む.

        Args:
            model_path: モデルディレクトリのパス.
            num_classes: クラス数 (背景クラスを含まない).
            pretrained: モデル構造を事前学習済みバックボーンに合わせるか.
                学習時に pretrained=True で作成したモデルの読み込みには True が必要.
            nms_iou_threshold: NMS IoU 閾値.
                ONNX エクスポートは NMS 前の生出力のみを含むため,
                エクスポート結果には影響しない.
        """
        self.model = SSDLiteModel(
            num_classes=num_classes,
            pretrained=pretrained,
            nms_iou_threshold=nms_iou_threshold,
        )
        self.model.load(model_path)
        self.model.to(self.device)
        logger.info(f"モデルの読み込み完了: {model_path}")

    def export(
        self,
        output_path: Path,
        input_size: tuple[int, int],
        opset_version: int = 17,
        fp16: bool = False,
    ) -> Path:
        """モデルを ONNX 形式でエクスポート.

        Args:
            output_path: 出力ファイルパス.
            input_size: 入力サイズ (height, width).
            opset_version: ONNX オペセットバージョン.
            fp16: FP16 でエクスポートするか.

        Returns:
            出力ファイルパス.

        Raises:
            ValueError: モデルが設定されていない場合.
        """
        wrapper, dummy_input = self._prepare_wrapper_and_input(input_size, fp16)
        if fp16:
            logger.info("FP16 モードでエクスポートします")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            torch.onnx.export(
                wrapper,
                (dummy_input,),
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["pixel_values"],
                output_names=["cls_logits", "bbox_regression"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
                    "cls_logits": {0: "batch_size"},
                    "bbox_regression": {0: "batch_size"},
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
        fp16: bool = False,
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ) -> bool:
        """エクスポートした ONNX モデルを検証.

        PyTorch モデルと ONNX モデルの出力を比較し,
        許容誤差内で一致するかを確認する.

        Args:
            onnx_path: ONNX モデルのパス.
            input_size: 入力サイズ (height, width).
            fp16: FP16 モデルかどうか.
            rtol: 相対許容誤差.
            atol: 絶対許容誤差.

        Returns:
            検証成功の場合 True.

        Raises:
            ValueError: モデルが設定されていない場合.
        """
        wrapper, dummy_input = self._prepare_wrapper_and_input(input_size, fp16)

        with torch.no_grad():
            pt_logits, pt_boxes = wrapper(dummy_input)

        pt_logits_np = pt_logits.float().cpu().numpy()
        pt_boxes_np = pt_boxes.float().cpu().numpy()

        # FP16 では数値精度の制約により許容誤差を緩める
        if fp16:
            rtol = max(rtol, 5e-2)
            atol = max(atol, 1e-1)

        return verify_onnx_outputs(
            onnx_path=onnx_path,
            pytorch_outputs=[pt_logits_np, pt_boxes_np],
            dummy_input=dummy_input.cpu().numpy(),
            output_names=["cls_logits", "bbox_regression"],
            rtol=rtol,
            atol=atol,
        )
