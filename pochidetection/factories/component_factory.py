"""コンポーネントファクトリー.

Factory + Registry パターンによる依存性注入.
設定辞書からコンポーネントを動的に生成する.
"""

from typing import Any, Callable

from pochidetection.interfaces import (
    IDetectionDataset,
    IDetectionLoss,
    IDetectionMetrics,
    IDetectionModel,
)

# ファクトリー関数の型エイリアス
ModelFactory = Callable[..., IDetectionModel]
LossFactory = Callable[..., IDetectionLoss]
MetricsFactory = Callable[..., IDetectionMetrics]
DatasetFactory = Callable[..., IDetectionDataset]


class ComponentFactory:
    """依存性注入用ファクトリー.

    DIP: 高レベルモジュール (Trainer) は抽象に依存.
    このファクトリーで具象コンポーネントを生成し, Trainerに注入する.

    Attributes:
        _model_registry: 登録されたモデルファクトリーの辞書.
        _loss_registry: 登録された損失関数ファクトリーの辞書.
        _metrics_registry: 登録された評価指標ファクトリーの辞書.
        _dataset_registry: 登録されたデータセットファクトリーの辞書.
    """

    _model_registry: dict[str, ModelFactory] = {}
    _loss_registry: dict[str, LossFactory] = {}
    _metrics_registry: dict[str, MetricsFactory] = {}
    _dataset_registry: dict[str, DatasetFactory] = {}

    @classmethod
    def register_model(cls, name: str, model_class: ModelFactory) -> None:
        """モデルを登録.

        Args:
            name: モデル名 (設定ファイルで使用).
            model_class: IDetectionModelを返すファクトリー (通常はクラス).
        """
        cls._model_registry[name] = model_class

    @classmethod
    def register_loss(cls, name: str, loss_class: LossFactory) -> None:
        """損失関数を登録.

        Args:
            name: 損失関数名 (設定ファイルで使用).
            loss_class: IDetectionLossを返すファクトリー (通常はクラス).
        """
        cls._loss_registry[name] = loss_class

    @classmethod
    def register_metrics(cls, name: str, metrics_class: MetricsFactory) -> None:
        """評価指標を登録.

        Args:
            name: 評価指標名 (設定ファイルで使用).
            metrics_class: IDetectionMetricsを返すファクトリー (通常はクラス).
        """
        cls._metrics_registry[name] = metrics_class

    @classmethod
    def register_dataset(cls, name: str, dataset_class: DatasetFactory) -> None:
        """データセットを登録.

        Args:
            name: データセット名 (設定ファイルで使用).
            dataset_class: IDetectionDatasetを返すファクトリー (通常はクラス).
        """
        cls._dataset_registry[name] = dataset_class

    @classmethod
    def create_model(cls, config: dict[str, Any]) -> IDetectionModel:
        """設定からモデルを生成.

        Args:
            config: モデル設定を含む辞書.
                - architecture: モデル名 (デフォルト: "RTDetr").
                - model_name: HuggingFaceモデル名 (デフォルト: "PekingU/rtdetr_r50vd").
                - num_classes: クラス数 (必須).
                - pretrained: 事前学習済み重みを使用するか (デフォルト: True).

        Returns:
            生成されたモデルインスタンス.

        Raises:
            ValueError: 未登録のモデル名が指定された場合.
            KeyError: 必須パラメータが設定に存在しない場合.
        """
        architecture = config.get("architecture", "RTDetr")
        if architecture not in cls._model_registry:
            available = list(cls._model_registry.keys())
            raise ValueError(f"未登録のモデル: {architecture}. 利用可能: {available}")

        return cls._model_registry[architecture](
            model_name=config.get("model_name", "PekingU/rtdetr_r50vd"),
            num_classes=config["num_classes"],
            pretrained=config.get("pretrained", True),
        )

    @classmethod
    def create_loss(cls, config: dict[str, Any]) -> IDetectionLoss:
        """設定から損失関数を生成.

        Args:
            config: 損失関数設定を含む辞書.
                - loss: 損失関数名 (デフォルト: "DetectionLoss").
                - loss_params: 損失関数に渡す追加パラメータ (デフォルト: {}).

        Returns:
            生成された損失関数インスタンス.

        Raises:
            ValueError: 未登録の損失関数名が指定された場合.
        """
        loss_name = config.get("loss", "DetectionLoss")
        if loss_name not in cls._loss_registry:
            available = list(cls._loss_registry.keys())
            raise ValueError(f"未登録の損失関数: {loss_name}. 利用可能: {available}")

        loss_params = config.get("loss_params", {})
        return cls._loss_registry[loss_name](**loss_params)

    @classmethod
    def create_metrics(cls, config: dict[str, Any]) -> IDetectionMetrics:
        """設定から評価指標を生成.

        Args:
            config: 評価指標設定を含む辞書.
                - metrics: 評価指標名 (デフォルト: "DetectionMetrics").
                - metrics_params: 評価指標に渡す追加パラメータ (デフォルト: {}).

        Returns:
            生成された評価指標インスタンス.

        Raises:
            ValueError: 未登録の評価指標名が指定された場合.
        """
        metrics_name = config.get("metrics", "DetectionMetrics")
        if metrics_name not in cls._metrics_registry:
            available = list(cls._metrics_registry.keys())
            raise ValueError(f"未登録の評価指標: {metrics_name}. 利用可能: {available}")

        metrics_params = config.get("metrics_params", {})
        return cls._metrics_registry[metrics_name](**metrics_params)

    @classmethod
    def create_dataset(cls, config: dict[str, Any]) -> IDetectionDataset:
        """設定からデータセットを生成.

        Args:
            config: データセット設定を含む辞書.
                - dataset: データセット名 (デフォルト: "CocoDetectionDataset").
                - image_dir: 画像ディレクトリパス (必須).
                - annotation_file: アノテーションファイルパス (オプション).

        Returns:
            生成されたデータセットインスタンス.

        Raises:
            ValueError: 未登録のデータセット名が指定された場合.
            KeyError: 必須パラメータが設定に存在しない場合.
        """
        dataset_name = config.get("dataset", "CocoDetectionDataset")
        if dataset_name not in cls._dataset_registry:
            available = list(cls._dataset_registry.keys())
            raise ValueError(
                f"未登録のデータセット: {dataset_name}. 利用可能: {available}"
            )

        return cls._dataset_registry[dataset_name](
            root=config["image_dir"],
            annotation_file=config.get("annotation_file"),
        )

    @classmethod
    def get_available_models(cls) -> list[str]:
        """登録済みモデル名のリストを取得.

        Returns:
            登録済みモデル名のリスト.
        """
        return list(cls._model_registry.keys())

    @classmethod
    def get_available_losses(cls) -> list[str]:
        """登録済み損失関数名のリストを取得.

        Returns:
            登録済み損失関数名のリスト.
        """
        return list(cls._loss_registry.keys())

    @classmethod
    def get_available_metrics(cls) -> list[str]:
        """登録済み評価指標名のリストを取得.

        Returns:
            登録済み評価指標名のリスト.
        """
        return list(cls._metrics_registry.keys())

    @classmethod
    def get_available_datasets(cls) -> list[str]:
        """登録済みデータセット名のリストを取得.

        Returns:
            登録済みデータセット名のリスト.
        """
        return list(cls._dataset_registry.keys())
