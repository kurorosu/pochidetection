"""SSD300 VGG16 学習スクリプト.

torchvision の SSD300 を COCO 形式データセットでファインチューニングする.
"""

import logging
from functools import partial

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.datasets import SsdCocoDataset
from pochidetection.interfaces.model import IDetectionModel
from pochidetection.logging import LoggerManager
from pochidetection.models import SSD300Model
from pochidetection.training.loop import (
    TrainingContext,
    TrainingLoop,
    setup_training,
)
from pochidetection.training.validation import ssd_validate


def train(config: DetectionConfigDict, config_path: str) -> None:
    """ファインチューニング.

    Args:
        config: 設定辞書.
        config_path: 設定ファイルのパス (ワークスペースにコピーするため).
    """
    logger = LoggerManager().get_logger(__name__)
    ctx = _setup_training(config, config_path, logger)
    TrainingLoop(config, ctx, ssd_validate).run()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _create_model(config: DetectionConfigDict) -> IDetectionModel:
    """モデル固有の SSD300 インスタンスを構築する.

    Args:
        config: 設定辞書.

    Returns:
        構築済みの SSD300Model.
    """
    num_classes = config["num_classes"]
    nms_iou_threshold = config.get("nms_iou_threshold", 0.5)
    return SSD300Model(num_classes=num_classes, nms_iou_threshold=nms_iou_threshold)


def _setup_training(
    config: DetectionConfigDict,
    config_path: str,
    logger: logging.Logger,
) -> TrainingContext:
    """学習環境の構築.

    Args:
        config: 設定辞書.
        config_path: 設定ファイルのパス.
        logger: ロガー.

    Returns:
        構築済みの学習コンテキスト.
    """
    logger.info("Architecture: SSD300 VGG16")

    image_size = config["image_size"]
    dataset_factory = partial(SsdCocoDataset, image_size=image_size)

    return setup_training(
        config=config,
        config_path=config_path,
        model_factory=_create_model,
        dataset_factory=dataset_factory,
        logger=logger,
    )
