"""学習ループの共通ロジック.

RT-DETR と SSDLite で共有されるエポックループ, Early Stopping,
データローダー構築, レポート出力のロジックを提供する.
"""

import logging
from collections.abc import Callable, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, cast

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import v2

from pochidetection.configs.schemas import AugmentationConfig, DetectionConfigDict
from pochidetection.core import DetectionCollator
from pochidetection.datasets.augmentation import build_augmentation
from pochidetection.datasets.base_coco_dataset import BaseCocoDataset
from pochidetection.interfaces.model import IDetectionModel
from pochidetection.logging import LoggerManager
from pochidetection.pipelines.builder import setup_cudnn_benchmark
from pochidetection.utils import (
    EarlyStopping,
    TrainingHistory,
    WorkspaceManager,
    build_scheduler,
)
from pochidetection.visualization import (
    F1ConfidencePlotter,
    LossPlotter,
    MetricsPlotter,
    PRCurvePlotter,
    TrainingReportPlotter,
)
from pochidetection.visualization.tensorboard import TensorBoardWriter


class TrainingContext(NamedTuple):
    """学習コンテキスト (アーキテクチャ共通)."""

    model: IDetectionModel
    train_loader: DataLoader[dict[str, Any]]
    val_loader: DataLoader[dict[str, Any]]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    map_metric: MeanAveragePrecision
    workspace: Path
    workspace_manager: WorkspaceManager
    device: str
    epochs: int
    train_score_threshold: float


DatasetFactory = Callable[[Path], Dataset[dict[str, Any]]]
ModelFactory = Callable[[DetectionConfigDict], IDetectionModel]


def _apply_augmentation_to_dataset(
    dataset: Dataset[dict[str, Any]],
    augmentation: v2.Compose,
) -> None:
    """データセットに augmentation を適用する.

    Args:
        dataset: 対象データセット.
        augmentation: 構築済みの augmentation パイプライン.
    """
    if not isinstance(dataset, BaseCocoDataset):
        return

    dataset._augmentation = augmentation


def _apply_debug_save_to_dataset(
    dataset: Dataset[dict[str, Any]],
    debug_save_count: int,
    debug_save_dir: Path,
    logger: logging.Logger | None = None,
) -> None:
    """データセットに学習画像デバッグ保存設定を適用する.

    augmentation の有無に関わらず, 1 エポック目の先頭 ``debug_save_count`` 枚を
    bbox 付きで保存する. letterbox / preprocess の silent bug を目視検知する目的.

    Args:
        dataset: 対象データセット.
        debug_save_count: 保存枚数 (> 0).
        debug_save_dir: 保存先ディレクトリ.
        logger: ロガー (None の場合はログ出力なし).
    """
    if not isinstance(dataset, BaseCocoDataset):
        return

    dataset.debug_save_count = debug_save_count
    dataset.debug_save_dir = debug_save_dir
    if logger is not None:
        logger.info(
            f"Train debug: saving first {debug_save_count} "
            f"images to {debug_save_dir}"
        )


def setup_training(
    config: DetectionConfigDict,
    config_path: str,
    model_factory: ModelFactory,
    dataset_factory: DatasetFactory,
    logger: logging.Logger,
) -> TrainingContext:
    """学習環境の共通セットアップ.

    ワークスペース作成, モデル構築, データローダー構築, オプティマイザ,
    スケジューラ, mAP メトリクスの初期化を共通化する.

    Args:
        config: 設定辞書.
        config_path: 設定ファイルのパス.
        model_factory: config を受け取りモデルを返すファクトリ.
        dataset_factory: ディレクトリパスを受け取りデータセットを返すファクトリ.
        logger: ロガー.

    Returns:
        構築済みの学習コンテキスト.
    """
    setup_cudnn_benchmark(config)

    device = config["device"]
    image_size = config.get("image_size", {"height": 640, "width": 640})
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    train_score_threshold = config["train_score_threshold"]

    workspace_manager = WorkspaceManager(config["work_dir"])
    workspace = workspace_manager.create_workspace()
    workspace_manager.save_config(config, Path(config_path).name)

    logger.info(f"Device: {device}")
    logger.info(f"Num classes: {config['num_classes']}")
    logger.info(f"Image size: {image_size}")
    logger.info(f"Workspace: {workspace}")

    model = model_factory(config)
    model.to(device)

    # augmentation 構築 (学習データのみに適用)
    aug_config = config.get("augmentation")
    augmentation = None
    if aug_config is not None:
        parsed = AugmentationConfig.model_validate(aug_config)
        augmentation = build_augmentation(parsed)

    # デバッグ画像保存 (augmentation の有無と独立)
    debug_save_count = config.get("debug_save_count", 0)

    train_loader, val_loader = build_data_loaders(config, dataset_factory, logger)

    # augmentation は学習データのみに適用
    if augmentation is not None:
        _apply_augmentation_to_dataset(train_loader.dataset, augmentation)

    # デバッグ画像保存も学習データのみに適用
    if debug_save_count > 0:
        _apply_debug_save_to_dataset(
            train_loader.dataset,
            debug_save_count=debug_save_count,
            debug_save_dir=workspace / "train_debug",
            logger=logger,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=config.get("lr_scheduler"),
        scheduler_params=config.get("lr_scheduler_params"),
        epochs=epochs,
    )
    if scheduler is not None:
        logger.info(f"LR Scheduler: {scheduler.__class__.__name__}")

    map_metric = MeanAveragePrecision(iou_type="bbox", extended_summary=True)
    map_metric.warn_on_many_detections = False

    return TrainingContext(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        map_metric=map_metric,
        workspace=workspace,
        workspace_manager=workspace_manager,
        device=device,
        epochs=epochs,
        train_score_threshold=train_score_threshold,
    )


def build_data_loaders(
    config: DetectionConfigDict,
    dataset_factory: DatasetFactory,
    logger: logging.Logger,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    """学習・検証用データローダーを構築.

    Args:
        config: 設定辞書.
        dataset_factory: ディレクトリパスを受け取りデータセットを返すファクトリ.
        logger: ロガー.

    Returns:
        (train_loader, val_loader) のタプル.

    Raises:
        ValueError: データローダーが空の場合.
    """
    data_root = Path(config["data_root"])
    train_dir = data_root / config["train_split"]
    val_dir = data_root / config["val_split"]
    batch_size = config["batch_size"]

    train_dataset = dataset_factory(train_dir)
    val_dataset = dataset_factory(val_dir)

    # ``torch.utils.data.Dataset`` は ``__len__`` を要求しないが, 本プロジェクトの
    # データセットは ``BaseCocoDataset`` 経由で ``__len__`` を実装している.
    # ``Sized`` にキャストして ``len()`` を型安全に呼び出す.
    logger.info(f"Train samples: {len(cast(Sized, train_dataset))}")
    logger.info(f"Val samples: {len(cast(Sized, val_dataset))}")

    collator = DetectionCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError(
            f"DataLoader が空です (train: {len(train_loader)} batches, "
            f"val: {len(val_loader)} batches). "
            f"データセットまたはバッチサイズを確認してください."
        )

    return train_loader, val_loader


class Validator(Protocol):
    """検証ループのプロトコル."""

    def __call__(
        self, ctx: TrainingContext, logger: logging.Logger
    ) -> tuple[float, dict[str, Any]]:
        """検証を実行して損失と mAP 結果を返す.

        Args:
            ctx: 学習コンテキスト.
            logger: ロガー.

        Returns:
            (平均検証損失, mAP 計算結果辞書) のタプル.
        """
        ...


@dataclass(frozen=True, slots=True)
class EpochResult:
    """1 エポックの学習・検証結果.

    Attributes:
        epoch: エポック番号 (1-indexed).
        train_loss: 平均学習損失.
        val_loss: 平均検証損失.
        map: Mean Average Precision.
        map_50: mAP at IoU=0.50.
        map_75: mAP at IoU=0.75.
        lr: 学習率.
        map_result: mAP 計算の生結果辞書.
    """

    epoch: int
    train_loss: float
    val_loss: float
    map: float
    map_50: float
    map_75: float
    lr: float
    map_result: dict[str, Any]


class TrainingLoop:
    """責務分離された学習ループ.

    エポックループのオーケストレーションを担当し,
    各責務をプライベートメソッドに委譲する.

    Args:
        config: 設定辞書.
        ctx: 学習コンテキスト.
        validate: 検証関数.
    """

    def __init__(
        self,
        config: DetectionConfigDict,
        ctx: TrainingContext,
        validate: Validator,
    ) -> None:
        """初期化."""
        self._config = config
        self._ctx = ctx
        self._validate = validate
        self._logger = LoggerManager().get_logger(__name__)
        self._history = TrainingHistory()
        self._best_map = 0.0

    def run(self) -> None:
        """学習ループを実行."""
        tb_writer = _setup_tensorboard(self._config, self._ctx.workspace, self._logger)
        early_stopping = build_early_stopping(self._config)
        if early_stopping is not None:
            self._log_early_stopping_config(early_stopping)

        last_map_result: dict[str, Any] = {}

        try:
            for epoch in range(self._ctx.epochs):
                result = self._run_one_epoch(epoch)
                last_map_result = result.map_result

                self._log_epoch(result)
                self._record_history(result)
                self._record_tensorboard(tb_writer, result)
                self._step_scheduler(result)

                if self._check_early_stopping(early_stopping, result):
                    break
        finally:
            if tb_writer is not None:
                tb_writer.close()

        save_results(
            self._config, self._ctx, self._history, last_map_result, self._logger
        )

    def _run_one_epoch(self, epoch: int) -> EpochResult:
        """1 エポックの学習と検証を実行.

        Args:
            epoch: エポックインデックス (0-indexed).

        Returns:
            エポック結果.
        """
        avg_loss, lr = train_one_epoch(self._ctx)
        avg_val_loss, map_result = self._validate(self._ctx, self._logger)

        return EpochResult(
            epoch=epoch + 1,
            train_loss=avg_loss,
            val_loss=avg_val_loss,
            map=map_result["map"].item(),
            map_50=map_result["map_50"].item(),
            map_75=map_result["map_75"].item(),
            lr=lr,
            map_result=map_result,
        )

    def _log_epoch(self, result: EpochResult) -> None:
        """エポック結果をログ出力.

        Args:
            result: エポック結果.
        """
        self._logger.info(
            f"Epoch {result.epoch}/{self._ctx.epochs} - "
            f"Train Loss: {result.train_loss:.4f}, "
            f"Val Loss: {result.val_loss:.4f}, "
            f"mAP: {result.map:.4f}, "
            f"mAP@50: {result.map_50:.4f}, "
            f"mAP@75: {result.map_75:.4f}, "
            f"LR: {result.lr:.2e}"
        )

    def _record_history(self, result: EpochResult) -> None:
        """エポック結果を履歴に追加.

        Args:
            result: エポック結果.
        """
        self._history.add(
            epoch=result.epoch,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            map=result.map,
            map_50=result.map_50,
            map_75=result.map_75,
            lr=result.lr,
        )

    def _record_tensorboard(
        self,
        tb_writer: TensorBoardWriter | None,
        result: EpochResult,
    ) -> None:
        """Record metrics to TensorBoard.

        Args:
            tb_writer: TensorBoard ライター (None なら何もしない).
            result: エポック結果.
        """
        if tb_writer is None:
            return
        tb_writer.record_epoch(
            epoch=result.epoch,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            map_value=result.map,
            map_50=result.map_50,
            map_75=result.map_75,
            lr=result.lr,
        )

    def _step_scheduler(self, result: EpochResult) -> None:
        """学習率スケジューラを更新.

        Args:
            result: エポック結果.
        """
        if self._ctx.scheduler is None:
            return
        if isinstance(self._ctx.scheduler, ReduceLROnPlateau):
            self._ctx.scheduler.step(result.val_loss)
        else:
            self._ctx.scheduler.step()

    def _check_early_stopping(
        self,
        early_stopping: EarlyStopping | None,
        result: EpochResult,
    ) -> bool:
        """Early Stopping を判定し, ベストモデルを保存.

        Args:
            early_stopping: EarlyStopping インスタンス (None なら mAP ベースで保存).
            result: エポック結果.

        Returns:
            True なら学習を停止すべき.
        """
        if early_stopping is None:
            if result.map > self._best_map:
                self._best_map = result.map
                _save_best(self._ctx, "mAP", result.map, self._logger)
            return False

        metric = self._config["early_stopping_metric"]
        value = result.map if metric == "mAP" else result.val_loss
        should_stop = early_stopping.step(value, result.epoch)

        if early_stopping.counter == 0:
            _save_best(self._ctx, metric, value, self._logger)

        if should_stop:
            self._logger.info(
                f"Early Stopping: {early_stopping.patience} エポック連続で "
                f"{metric} が改善しなかったため学習を終了します "
                f"(best epoch: {early_stopping.best_epoch}, "
                f"best {metric}: {early_stopping.best_value:.4f})"
            )

        return should_stop

    def _log_early_stopping_config(self, early_stopping: EarlyStopping) -> None:
        """Early Stopping の設定をログ出力.

        Args:
            early_stopping: EarlyStopping インスタンス.
        """
        self._logger.info(
            f"Early Stopping: patience={early_stopping.patience}, "
            f"metric={self._config['early_stopping_metric']}, "
            f"min_delta={self._config['early_stopping_min_delta']}"
        )


def build_early_stopping(config: DetectionConfigDict) -> EarlyStopping | None:
    """設定から EarlyStopping を構築.

    Args:
        config: 設定辞書.

    Returns:
        EarlyStopping インスタンス, 無効時は None.
    """
    patience = config.get("early_stopping_patience")
    if patience is None:
        return None

    metric = config["early_stopping_metric"]
    mode: Literal["min", "max"] = "max" if metric == "mAP" else "min"
    min_delta = config["early_stopping_min_delta"]

    return EarlyStopping(patience=patience, min_delta=min_delta, mode=mode)


def train_one_epoch(ctx: TrainingContext) -> tuple[float, float]:
    """1エポックの学習.

    Args:
        ctx: 学習コンテキスト.

    Returns:
        (平均学習損失, 現在の学習率) のタプル.
    """
    ctx.model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(ctx.train_loader):
        pixel_values = batch["pixel_values"].to(ctx.device)
        labels = [{k: v.to(ctx.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = ctx.model(pixel_values=pixel_values, labels=labels)
        loss = outputs["loss"]

        ctx.optimizer.zero_grad()
        loss.backward()
        ctx.optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(ctx.train_loader)
    lr = ctx.optimizer.param_groups[0]["lr"]
    return avg_loss, lr


def save_results(
    config: DetectionConfigDict,
    ctx: TrainingContext,
    history: TrainingHistory,
    map_result: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """モデル保存 + レポート出力.

    Args:
        config: 設定辞書.
        ctx: 学習コンテキスト.
        history: 学習履歴.
        map_result: 最後に検証されたエポックの mAP 計算結果.
        logger: ロガー.
    """
    last_dir = ctx.workspace_manager.get_last_dir()
    ctx.model.save(last_dir)
    logger.info(f"Last model saved to {last_dir}")

    history_path = ctx.workspace / "training_history.csv"
    history.save_csv(history_path)
    logger.info(f"Training history saved to {history_path}")

    loss_plotter = LossPlotter(history)
    metrics_plotter = MetricsPlotter(history)
    report_plotter = TrainingReportPlotter(loss_plotter, metrics_plotter)
    report_path = ctx.workspace / "training_report.html"
    report_plotter.plot(report_path)
    logger.info(f"Training report saved to {report_path}")

    class_names = config.get("class_names")

    if "precision" in map_result:
        pr_plotter = PRCurvePlotter(map_result["precision"], class_names=class_names)
        pr_path = ctx.workspace / "pr_curve.html"
        pr_plotter.plot(pr_path)
        logger.info(f"PR curve saved to {pr_path}")

    if "precision" in map_result and "scores" in map_result:
        f1_plotter = F1ConfidencePlotter(
            map_result["precision"],
            map_result["scores"],
            class_names=class_names,
        )
        f1_path = ctx.workspace / "f1_confidence.html"
        f1_plotter.plot(f1_path)
        logger.info(f"F1-Confidence curve saved to {f1_path}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _setup_tensorboard(
    config: DetectionConfigDict,
    workspace: Path,
    logger: logging.Logger,
) -> TensorBoardWriter | None:
    """設定に応じて TensorBoard ライターを初期化.

    Args:
        config: 設定辞書.
        workspace: ワークスペースディレクトリ.
        logger: ロガー.

    Returns:
        TensorBoardWriter インスタンス, 無効時は None.
    """
    if not config.get("enable_tensorboard", False):
        return None

    workspace_name = workspace.name
    tb_log_dir = workspace / "tensorboard" / workspace_name
    writer = TensorBoardWriter(log_dir=tb_log_dir, logger=logger)
    logger.info(f"TensorBoard: tensorboard --logdir {tb_log_dir.parent}")
    return writer


def _save_best(
    ctx: TrainingContext,
    metric_name: str,
    metric_value: float,
    logger: logging.Logger,
) -> None:
    """Best model を保存.

    Args:
        ctx: 学習コンテキスト.
        metric_name: メトリクス名 (ログ用).
        metric_value: メトリクス値 (ログ用).
        logger: ロガー.
    """
    best_dir = ctx.workspace_manager.get_best_dir()
    ctx.model.save(best_dir)
    logger.info(f"Best model saved to {best_dir} ({metric_name}: {metric_value:.4f})")
