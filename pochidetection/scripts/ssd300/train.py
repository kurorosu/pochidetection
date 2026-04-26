"""SSD300 VGG16 学習スクリプト.

torchvision の SSD300 を COCO 形式データセットでファインチューニングする.
共通ロジックは :func:`pochidetection.training.ssd.train_ssd` に集約されている.
"""

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.models import SSD300Model
from pochidetection.training.ssd import train_ssd


def train(config: DetectionConfigDict, config_path: str) -> None:
    """ファインチューニング.

    Args:
        config: 設定辞書.
        config_path: 設定ファイルのパス (ワークスペースにコピーするため).
    """
    train_ssd(config, config_path, SSD300Model, "SSD300 VGG16")
