"""SSDLite MobileNetV3 物体検出モデル."""

from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSD

from pochidetection.models.ssd_base import SSDModelBase


class SSDLiteModel(SSDModelBase):
    """SSDLite MobileNetV3 モデルのラッパー.

    torchvision の ssdlite320_mobilenet_v3_large をラップし,
    IDetectionModel インターフェースを実装する. 共通ロジック
    (``forward`` / ``save`` / ``load`` / ラベル変換) は ``SSDModelBase`` に
    集約されている.

    Note:
        NMS は torchvision の SSD 内部 (``postprocess_detections``) で
        自動適用される. ``nms_iou_threshold`` でその閾値を制御できる.
        推論パイプライン側で明示的に NMS を呼ぶ必要はない.
        RT-DETR が ``torchvision.ops.nms`` を後処理で明示適用するのとは
        設計が異なる.

    Attributes:
        _model: torchvision の SSD モデルインスタンス.
        _num_classes: ユーザ指定のクラス数 (背景なし).
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        nms_iou_threshold: float = 0.55,
    ) -> None:
        """初期化.

        Args:
            num_classes: クラス数 (背景クラスを含まない).
            pretrained: 事前学習済みバックボーン重みを使用するかどうか.
            nms_iou_threshold: NMS の IoU 閾値. torchvision の nms_thresh に渡される.
        """
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            nms_iou_threshold=nms_iou_threshold,
        )

    def _create_torchvision_model(
        self,
        num_classes: int,
        nms_iou_threshold: float,
        pretrained: bool,
    ) -> SSD:
        """Torchvision の ssdlite320_mobilenet_v3_large を生成する."""
        weights_backbone = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        return ssdlite320_mobilenet_v3_large(
            weights_backbone=weights_backbone,
            num_classes=num_classes,
            nms_thresh=nms_iou_threshold,
        )
