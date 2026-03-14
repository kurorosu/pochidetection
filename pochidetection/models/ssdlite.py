"""SSDLite MobileNetV3 物体検出モデル."""

from pathlib import Path
from typing import Any

import torch
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSD

from pochidetection.interfaces.model import IDetectionModel


class SSDLiteModel(IDetectionModel):
    """SSDLite MobileNetV3 モデルのラッパー.

    torchvision の ssdlite320_mobilenet_v3_large をラップし,
    IDetectionModel インターフェースを実装する.

    SSD は背景クラス (label=0) を内部で使用するため,
    torchvision に渡す num_classes は ユーザ指定値 + 1 となる.
    label オフセット (+1/-1) は本クラスに集約し,
    外部との入出力は全て 0-indexed で統一する.

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
        super().__init__()

        # SSD は背景クラスを含むため +1
        ssd_num_classes = num_classes + 1

        weights_backbone = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self._model: SSD = ssdlite320_mobilenet_v3_large(
            weights_backbone=weights_backbone,
            num_classes=ssd_num_classes,
            nms_thresh=nms_iou_threshold,
        )
        self._num_classes = num_classes

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        """順伝播.

        Args:
            pixel_values: 入力画像テンソル, 形状は (B, C, H, W).
            labels: 学習時のターゲット. 各要素は以下のキーを含む辞書:
                - boxes: バウンディングボックス (N, 4), xyxy ピクセル座標
                - class_labels: クラスラベル (N,), 0-indexed

        Returns:
            以下のキーを含む辞書:
            - loss: 学習時の損失 (labels が指定された場合)
            - predictions: 推論時の検出結果 (labels が None の場合).
                list[dict] で各要素は boxes (M, 4), scores (M,),
                labels (M,) を含む (0-indexed).
        """
        images = list(pixel_values.unbind(0))

        if self._model.training and labels is not None:
            targets = [
                {
                    "boxes": t["boxes"],
                    "labels": t["class_labels"] + 1,  # 0-indexed → 1-indexed
                }
                for t in labels
            ]
            losses = self._model(images, targets)
            return {"loss": sum(losses.values())}

        detections = self._model(images)
        predictions = []
        for det in detections:
            raw_labels = det["labels"] - 1  # 1-indexed → 0-indexed
            # 背景クラス (torchvision label=0 → -1) を除去
            fg_mask = raw_labels >= 0
            predictions.append(
                {
                    "boxes": det["boxes"][fg_mask],
                    "scores": det["scores"][fg_mask],
                    "labels": raw_labels[fg_mask],
                }
            )
        return {"predictions": predictions}

    def save(self, save_dir: str | Path) -> None:
        """モデルを state_dict 形式で保存.

        Args:
            save_dir: 保存先ディレクトリパス.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), save_dir / "model.pth")

    def load(self, load_dir: str | Path) -> None:
        """state_dict 形式のディレクトリからモデルを復元.

        Args:
            load_dir: 読み込み元ディレクトリパス.
        """
        state_dict = torch.load(
            Path(load_dir) / "model.pth",
            map_location="cpu",
            weights_only=True,
        )
        self._model.load_state_dict(state_dict)

    @property
    def num_classes(self) -> int:
        """クラス数を取得 (背景クラスを含まない).

        Returns:
            クラス数.
        """
        return self._num_classes

    @property
    def model(self) -> torch.nn.Module:
        """内部モデルを取得.

        Returns:
            torchvision の SSD モデルインスタンス.
        """
        return self._model
