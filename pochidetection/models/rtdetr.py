"""RT-DETR物体検出モデル."""

from pathlib import Path
from typing import Any

import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from pochidetection.configs.schemas import ImageSizeDict
from pochidetection.interfaces.model import IDetectionModel, ModelOutputDict


class RTDetrModel(IDetectionModel):
    """RT-DETRモデルのラッパー.

    HuggingFace transformersのRTDetrForObjectDetectionをラップし,
    IDetectionModelインターフェースを実装する.

    Attributes:
        _model: transformersのRTDetrForObjectDetectionインスタンス.
        _processor: 画像前処理プロセッサ.
        _num_classes: クラス数.
    """

    def __init__(
        self,
        model_name: str = "PekingU/rtdetr_r50vd",
        num_classes: int | None = None,
        pretrained: bool = True,
        image_size: ImageSizeDict | None = None,
    ) -> None:
        """RTDetrModelを初期化.

        Args:
            model_name: HuggingFaceモデル名またはローカルパス.
            num_classes: クラス数. Noneの場合は事前学習済みモデルの設定を使用.
            pretrained: 事前学習済み重みを使用するかどうか.
            image_size: 画像サイズ {"height": int, "width": int}.
                Noneの場合はモデルのデフォルト設定を使用.
        """
        super().__init__()

        if pretrained:
            kwargs: dict[str, Any] = {}
            if num_classes is not None:
                kwargs["num_labels"] = num_classes
                kwargs["ignore_mismatched_sizes"] = True
            self._model = RTDetrForObjectDetection.from_pretrained(model_name, **kwargs)
        else:
            from transformers import RTDetrConfig

            config = RTDetrConfig.from_pretrained(model_name)
            if num_classes is not None:
                config.num_labels = num_classes
            self._model = RTDetrForObjectDetection(config)

        self._num_classes = num_classes or self._model.config.num_labels

        proc_kwargs: dict[str, Any] = {}
        if image_size is not None:
            proc_kwargs["size"] = image_size
        self._processor = RTDetrImageProcessor.from_pretrained(
            model_name, **proc_kwargs
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
    ) -> ModelOutputDict:
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

        result: ModelOutputDict = {
            "pred_logits": outputs.logits,
            "pred_boxes": outputs.pred_boxes,
        }

        if outputs.loss is not None:
            result["loss"] = outputs.loss

        return result

    def save(self, save_dir: str | Path) -> None:
        """モデルと processor を save_pretrained 形式で保存.

        Args:
            save_dir: 保存先ディレクトリパス.
        """
        self._model.save_pretrained(save_dir)
        self._processor.save_pretrained(save_dir)

    def load(self, load_dir: str | Path) -> None:
        """save_pretrained 形式のディレクトリからモデルと processor を復元.

        Args:
            load_dir: 読み込み元ディレクトリパス.
        """
        self._model = RTDetrForObjectDetection.from_pretrained(load_dir)
        self._processor = RTDetrImageProcessor.from_pretrained(load_dir)
        self._num_classes = self._model.config.num_labels

    @property
    def num_classes(self) -> int:
        """クラス数を取得.

        Returns:
            クラス数.
        """
        return self._num_classes

    @property
    def processor(self) -> RTDetrImageProcessor:
        """画像前処理プロセッサを取得.

        Returns:
            RTDetrImageProcessorインスタンス.
        """
        return self._processor

    @property
    def model(self) -> RTDetrForObjectDetection:
        """内部モデルを取得.

        Returns:
            transformersのRTDetrForObjectDetectionインスタンス.
        """
        return self._model
