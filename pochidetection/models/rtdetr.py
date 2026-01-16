"""RT-DETR物体検出モデル."""

from typing import Any

import torch
import torch.nn as nn
from transformers import RTDetrForObjectDetection

from pochidetection.interfaces.model import IDetectionModel


class RTDetrModel(IDetectionModel):
    """RT-DETRモデルのラッパー.

    HuggingFace transformersのRTDetrForObjectDetectionをラップし,
    IDetectionModelインターフェースを実装する.

    Attributes:
        _model: transformersのRTDetrForObjectDetectionインスタンス.
        _num_classes: クラス数.
    """

    def __init__(
        self,
        model_name: str = "PekingU/rtdetr_r50vd",
        num_classes: int | None = None,
        pretrained: bool = True,
    ) -> None:
        """RTDetrModelを初期化.

        Args:
            model_name: HuggingFaceモデル名またはローカルパス.
            num_classes: クラス数. Noneの場合は事前学習済みモデルの設定を使用.
            pretrained: 事前学習済み重みを使用するかどうか.
        """
        super().__init__()

        if pretrained:
            self._model = RTDetrForObjectDetection.from_pretrained(model_name)
        else:
            from transformers import RTDetrConfig

            config = RTDetrConfig.from_pretrained(model_name)
            if num_classes is not None:
                config.num_labels = num_classes
            self._model = RTDetrForObjectDetection(config)

        # クラス数を更新 (事前学習済みモデルの場合も上書き可能)
        if num_classes is not None and pretrained:
            self._update_num_classes(num_classes)

        self._num_classes = num_classes or self._model.config.num_labels

    def _update_num_classes(self, num_classes: int) -> None:
        """クラス数を更新し, 分類ヘッドを再初期化.

        Args:
            num_classes: 新しいクラス数.
        """
        self._model.config.num_labels = num_classes

        # 分類ヘッドを再初期化
        hidden_size = self._model.config.d_model
        self._model.class_embed = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        """順伝播.

        Args:
            pixel_values: 入力画像テンソル, 形状は (B, C, H, W).
            labels: 学習時のターゲット. 各要素は以下のキーを含む辞書:
                - boxes: バウンディングボックス (N, 4), 正規化座標 [cx, cy, w, h]
                - class_labels: クラスラベル (N,)

        Returns:
            以下のキーを含む辞書:
            - loss: 学習時の損失 (labelsが指定された場合)
            - pred_boxes: 予測ボックス (B, num_queries, 4)
            - pred_logits: 予測ロジット (B, num_queries, num_classes)
        """
        outputs = self._model(pixel_values=pixel_values, labels=labels)

        result: dict[str, Any] = {
            "pred_logits": outputs.logits,
            "pred_boxes": outputs.pred_boxes,
        }

        if outputs.loss is not None:
            result["loss"] = outputs.loss

        return result

    def get_backbone_params(self) -> list[nn.Parameter]:
        """層別学習率用のバックボーンパラメータを取得.

        Returns:
            バックボーンパラメータのリスト.
        """
        backbone_params: list[nn.Parameter] = []
        for name, param in self._model.named_parameters():
            if "backbone" in name or "encoder" in name:
                backbone_params.append(param)
        return backbone_params

    def get_head_params(self) -> list[nn.Parameter]:
        """層別学習率用のヘッドパラメータを取得.

        Returns:
            ヘッドパラメータのリスト.
        """
        head_params: list[nn.Parameter] = []
        for name, param in self._model.named_parameters():
            if "backbone" not in name and "encoder" not in name:
                head_params.append(param)
        return head_params

    @property
    def num_classes(self) -> int:
        """クラス数を取得.

        Returns:
            クラス数.
        """
        return self._num_classes

    @property
    def model(self) -> RTDetrForObjectDetection:
        """内部モデルを取得.

        Returns:
            transformersのRTDetrForObjectDetectionインスタンス.
        """
        return self._model
