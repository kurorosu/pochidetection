"""SSDLite MobileNetV3 物体検出モデル."""

from pathlib import Path
from typing import Any

import torch
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from pochidetection.interfaces.model import IDetectionModel


class SSDLiteModel(IDetectionModel):
    """SSDLite MobileNetV3 モデルのラッパー.

    torchvision の ssdlite320_mobilenet_v3_large をラップし,
    IDetectionModel インターフェースを実装する.

    SSD は背景クラス (label=0) を内部で使用するため,
    torchvision に渡す num_classes は ユーザ指定値 + 1 となる.
    forward の出力では label を -1 して 0-indexed に戻す.

    Attributes:
        _model: torchvision の SSD モデルインスタンス.
        _num_classes: ユーザ指定のクラス数 (背景なし).
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        model_path: str | Path | None = None,
    ) -> None:
        """初期化.

        Args:
            num_classes: クラス数 (背景クラスを含まない).
            pretrained: 事前学習済みバックボーン重みを使用するかどうか.
            model_path: 保存済みモデルのパス. 指定時は state_dict をロードする.
        """
        super().__init__()

        # SSD は背景クラスを含むため +1
        ssd_num_classes = num_classes + 1

        weights_backbone = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self._model = ssdlite320_mobilenet_v3_large(
            weights_backbone=weights_backbone,
            num_classes=ssd_num_classes,
        )
        self._num_classes = num_classes

        if model_path is not None:
            state_dict = torch.load(
                Path(model_path) / "model.pth",
                map_location="cpu",
                weights_only=True,
            )
            self._model.load_state_dict(state_dict)

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
                - class_labels: クラスラベル (N,), 1-indexed (背景=0)

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
                {"boxes": t["boxes"], "labels": t["class_labels"]} for t in labels
            ]
            losses = self._model(images, targets)
            return {"loss": sum(losses.values())}

        detections = self._model(images)
        predictions = []
        for det in detections:
            predictions.append(
                {
                    "boxes": det["boxes"],
                    "scores": det["scores"],
                    "labels": det["labels"] - 1,  # 0-indexed に戻す
                }
            )
        return {"predictions": predictions}

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
        return self._model  # type: ignore[no-any-return]
