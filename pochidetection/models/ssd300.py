"""SSD300 VGG16 物体検出モデル."""

from torchvision.models import VGG16_Weights
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD

from pochidetection.models.ssd_base import SSDModelBase


class SSD300Model(SSDModelBase):
    """SSD300 VGG16 モデルのラッパー.

    torchvision の ssd300_vgg16 をラップし, IDetectionModel インターフェースを
    実装する. 共通ロジック (``__init__`` / ``forward`` / ``save`` / ``load`` /
    ラベル変換) は ``SSDModelBase`` に集約されている.

    Note:
        NMS は torchvision の SSD 内部 (``postprocess_detections``) で
        自動適用される. ``nms_iou_threshold`` でその閾値を制御できる
        (default 0.5).

    Attributes:
        _model: torchvision の SSD モデルインスタンス.
        _num_classes: ユーザ指定のクラス数 (背景なし).
    """

    def _create_torchvision_model(
        self,
        num_classes: int,
        nms_iou_threshold: float,
        pretrained: bool,
    ) -> SSD:
        """Torchvision の ssd300_vgg16 を生成する."""
        weights_backbone = VGG16_Weights.DEFAULT if pretrained else None
        return ssd300_vgg16(
            weights_backbone=weights_backbone,
            num_classes=num_classes,
            nms_thresh=nms_iou_threshold,
        )
