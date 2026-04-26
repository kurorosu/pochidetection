"""SSDLite MobileNetV3 学習スクリプト.

torchvision の SSDLite を COCO 形式データセットでファインチューニングする.
共通ロジックは :func:`pochidetection.training.ssd.train_ssd` に集約されている.
"""

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.models import SSDLiteModel
from pochidetection.training.ssd import train_ssd


def train(config: DetectionConfigDict, config_path: str) -> None:
    """ファインチューニング.

    Args:
        config: 設定辞書.
        config_path: 設定ファイルのパス (ワークスペースにコピーするため).
    """
    train_ssd(config, config_path, SSDLiteModel, "SSDLite MobileNetV3")
