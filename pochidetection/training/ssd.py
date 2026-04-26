"""SSD 系モデル (SSD300 / SSDLite) の共通学習ヘルパー.

両アーキの ``_setup_training`` がモデルクラスとアーキ名以外で完全に同一だったため,
共通フローを ``train_ssd`` にまとめ, 各スクリプトはモデルクラスとラベルだけを渡す.
"""

from functools import partial

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.datasets import SsdCocoDataset
from pochidetection.interfaces.model import IDetectionModel
from pochidetection.logging import LoggerManager
from pochidetection.training.loop import TrainingLoop, setup_training
from pochidetection.training.validation import ssd_validate

__all__ = ["train_ssd"]


def train_ssd(
    config: DetectionConfigDict,
    config_path: str,
    model_cls: type[IDetectionModel],
    arch_label: str,
) -> None:
    """SSD 系モデルの共通学習ループ.

    Args:
        config: 設定辞書.
        config_path: 設定ファイルのパス (ワークスペースにコピーするため).
        model_cls: 構築するモデルクラス (``SSD300Model`` / ``SSDLiteModel``).
            ``num_classes`` と ``nms_iou_threshold`` をキーワード引数で受け付ける.
        arch_label: ログ出力用のアーキテクチャ名 (例: ``"SSDLite MobileNetV3"``).
    """
    logger = LoggerManager().get_logger(__name__)
    logger.info(f"Architecture: {arch_label}")

    def model_factory(cfg: DetectionConfigDict) -> IDetectionModel:
        return model_cls(
            num_classes=cfg["num_classes"],
            nms_iou_threshold=cfg.get("nms_iou_threshold", 0.5),
        )

    image_size = config["image_size"]
    dataset_factory = partial(
        SsdCocoDataset,
        image_size=image_size,
        letterbox=config.get("letterbox", True),
    )

    ctx = setup_training(
        config=config,
        config_path=config_path,
        model_factory=model_factory,
        dataset_factory=dataset_factory,
        logger=logger,
    )
    TrainingLoop(config, ctx, ssd_validate).run()
