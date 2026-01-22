"""物体検出用Trainer.

pochisegmentationの設計思想に基づいた学習ループの実装.
DIP (Dependency Inversion Principle) に従い, インターフェースに依存する.
"""

from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from pochidetection.interfaces import (
    IDetectionLoss,
    IDetectionMetrics,
    IDetectionModel,
)
from pochidetection.logging import LoggerManager, LogLevel


class DetectionTrainer:
    """物体検出用Trainer.

    DIPに従い, インターフェースに依存して学習ループを実行する.
    AMP (Automatic Mixed Precision) をサポートし, チェックポイント保存を行う.

    Attributes:
        _model: 物体検出モデル (IDetectionModel).
        _loss_fn: 損失関数 (IDetectionLoss).
        _metrics: 評価指標 (IDetectionMetrics).
        _optimizer: オプティマイザ.
        _scheduler: 学習率スケジューラ (オプション).
        _device: デバイス.
        _work_dir: ワークスペースディレクトリ.
        _use_amp: AMPを使用するかどうか.
        _logger: ロガー.
    """

    def __init__(
        self,
        model: IDetectionModel,
        loss_fn: IDetectionLoss,
        metrics: IDetectionMetrics,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        device: str = "cuda",
        work_dir: str | Path = "work_dirs",
        use_amp: bool = True,
    ) -> None:
        """DetectionTrainerを初期化.

        Args:
            model: 物体検出モデル.
            loss_fn: 損失関数.
            metrics: 評価指標.
            optimizer: オプティマイザ.
            scheduler: 学習率スケジューラ. Noneの場合は使用しない.
            device: デバイス ("cuda" または "cpu").
            work_dir: ワークスペースディレクトリ.
            use_amp: AMPを使用するかどうか (CUDAのみ有効).
        """
        self._model = model.to(device)
        self._loss_fn = loss_fn
        self._metrics = metrics
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._work_dir = Path(work_dir)
        self._use_amp = use_amp and device == "cuda"

        # AMP用のスケーラー
        self._scaler = GradScaler() if self._use_amp else None

        # ロガー
        logger_manager = LoggerManager()
        self._logger = logger_manager.get_logger(__name__, level=LogLevel.INFO)

        # ワークスペースディレクトリを作成
        self._work_dir.mkdir(parents=True, exist_ok=True)

        # 学習履歴
        self._history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "mAP": [],
            "mAP_50": [],
            "mAP_75": [],
        }

        # ベストスコア
        self._best_map: float = 0.0

    def train(
        self,
        train_loader: DataLoader[dict[str, Any]],
        val_loader: DataLoader[dict[str, Any]],
        epochs: int,
        save_interval: int = 10,
    ) -> dict[str, list[float]]:
        """学習を実行.

        Args:
            train_loader: 学習データローダー.
            val_loader: 検証データローダー.
            epochs: エポック数.
            save_interval: チェックポイント保存間隔 (エポック).

        Returns:
            学習履歴の辞書.
        """
        self._logger.info(f"学習開始: {epochs} epochs, device={self._device}")
        self._logger.info(f"AMP: {'有効' if self._use_amp else '無効'}")

        for epoch in range(1, epochs + 1):
            # 学習
            train_loss = self._train_epoch(train_loader, epoch)
            self._history["train_loss"].append(train_loss)

            # 検証
            val_loss, val_metrics = self._validate(val_loader)
            self._history["val_loss"].append(val_loss)
            self._history["mAP"].append(val_metrics["mAP"])
            self._history["mAP_50"].append(val_metrics["mAP_50"])
            self._history["mAP_75"].append(val_metrics["mAP_75"])

            # ログ出力
            lr = self._optimizer.param_groups[0]["lr"]
            self._logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"train_loss: {train_loss:.4f}, "
                f"val_loss: {val_loss:.4f}, "
                f"mAP: {val_metrics['mAP']:.4f}, "
                f"mAP_50: {val_metrics['mAP_50']:.4f}, "
                f"lr: {lr:.2e}"
            )

            # スケジューラを更新
            if self._scheduler is not None:
                self._scheduler.step()

            # ベストモデルを保存
            if val_metrics["mAP"] > self._best_map:
                self._best_map = val_metrics["mAP"]
                self._save_checkpoint(epoch, is_best=True)
                self._logger.info(f"ベストモデルを保存: mAP={self._best_map:.4f}")

            # 定期的にチェックポイントを保存
            if epoch % save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

        # 最終モデルを保存
        self._save_checkpoint(epochs, is_best=False, filename="last.pth")
        self._logger.info("学習完了")

        return self._history

    def _train_epoch(
        self,
        train_loader: DataLoader[dict[str, Any]],
        epoch: int,
    ) -> float:
        """1エポックの学習を実行.

        Args:
            train_loader: 学習データローダー.
            epoch: 現在のエポック番号.

        Returns:
            平均学習損失.
        """
        self._model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            # データをデバイスに転送
            pixel_values = batch["pixel_values"].to(self._device)
            labels = self._prepare_labels(batch)

            # 勾配をリセット
            self._optimizer.zero_grad()

            # 順伝播 (AMP対応)
            if self._use_amp and self._scaler is not None:
                with autocast():
                    outputs = self._model(pixel_values, labels=labels)
                    loss = outputs.get("loss")
                    if loss is None:
                        loss = self._loss_fn(outputs, labels)

                # 逆伝播 (AMP)
                self._scaler.scale(loss).backward()
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                outputs = self._model(pixel_values, labels=labels)
                loss = outputs.get("loss")
                if loss is None:
                    loss = self._loss_fn(outputs, labels)

                # 逆伝播
                loss.backward()
                self._optimizer.step()

            total_loss += loss.item()

            # 進捗ログ (10バッチごと)
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                self._logger.debug(
                    f"Epoch {epoch} [{batch_idx + 1}/{num_batches}] - "
                    f"loss: {avg_loss:.4f}"
                )

        return total_loss / num_batches

    def _validate(
        self,
        val_loader: DataLoader[dict[str, Any]],
    ) -> tuple[float, dict[str, float]]:
        """検証を実行.

        Args:
            val_loader: 検証データローダー.

        Returns:
            (平均検証損失, 評価指標の辞書) のタプル.
        """
        self._model.eval()
        self._metrics.reset()
        total_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                # データをデバイスに転送
                pixel_values = batch["pixel_values"].to(self._device)
                labels = self._prepare_labels(batch)

                # 順伝播
                if self._use_amp:
                    with autocast():
                        outputs = self._model(pixel_values, labels=labels)
                else:
                    outputs = self._model(pixel_values, labels=labels)

                # 損失を計算
                loss = outputs.get("loss")
                if loss is None:
                    loss = self._loss_fn(outputs, labels)
                total_loss += loss.item()

                # 予測結果を後処理
                pred_boxes, pred_scores, pred_labels = self._postprocess_predictions(
                    outputs
                )

                # ターゲットを取得
                target_boxes = [label["boxes"] for label in labels]
                target_labels = [label["class_labels"] for label in labels]

                # 指標を更新
                self._metrics.update(
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_labels=pred_labels,
                    target_boxes=target_boxes,
                    target_labels=target_labels,
                )

        # 指標を計算
        metrics = self._metrics.compute()

        return total_loss / num_batches, metrics

    def _prepare_labels(
        self,
        batch: dict[str, Any],
    ) -> list[dict[str, torch.Tensor]]:
        """バッチからラベルを準備.

        Args:
            batch: データローダーからのバッチ.

        Returns:
            RT-DETR形式のラベルリスト.
        """
        labels = []
        batch_size = batch["pixel_values"].shape[0]

        for i in range(batch_size):
            label = {
                "boxes": batch["boxes"][i].to(self._device),
                "class_labels": batch["labels"][i].to(self._device),
            }
            labels.append(label)

        return labels

    def _postprocess_predictions(
        self,
        outputs: dict[str, Any],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """モデル出力から予測結果を後処理.

        Args:
            outputs: モデルの出力.

        Returns:
            (pred_boxes, pred_scores, pred_labels) のタプル.
            各要素はバッチサイズ分のリスト.
        """
        pred_logits = outputs["pred_logits"]  # (B, num_queries, num_classes)
        pred_boxes = outputs["pred_boxes"]  # (B, num_queries, 4)

        batch_size = pred_logits.shape[0]

        all_boxes = []
        all_scores = []
        all_labels = []

        for i in range(batch_size):
            logits = pred_logits[i]  # (num_queries, num_classes)
            boxes = pred_boxes[i]  # (num_queries, 4)

            # 確率に変換
            probs = logits.softmax(dim=-1)

            # 最大クラスとスコアを取得
            scores, labels = probs.max(dim=-1)

            # cxcywh -> xyxy (相対座標のまま)
            # RT-DETRの出力は [cx, cy, w, h] 形式 (0-1正規化)
            cx, cy, w, h = boxes.unbind(-1)
            xyxy_boxes = torch.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
            )

            all_boxes.append(xyxy_boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool,
        filename: str | None = None,
    ) -> None:
        """チェックポイントを保存.

        Args:
            epoch: 現在のエポック番号.
            is_best: ベストモデルかどうか.
            filename: 保存ファイル名. Noneの場合は自動生成.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "best_map": self._best_map,
            "history": self._history,
        }

        if self._scheduler is not None:
            checkpoint["scheduler_state_dict"] = self._scheduler.state_dict()

        if self._scaler is not None:
            checkpoint["scaler_state_dict"] = self._scaler.state_dict()

        # ファイル名を決定
        if filename is None:
            if is_best:
                filename = "best.pth"
            else:
                filename = f"epoch_{epoch:03d}.pth"

        # 保存
        save_path = self._work_dir / filename
        torch.save(checkpoint, save_path)
        self._logger.debug(f"チェックポイントを保存: {save_path}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """チェックポイントを読み込み.

        Args:
            checkpoint_path: チェックポイントファイルのパス.

        Returns:
            読み込んだエポック番号.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self._device)

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._best_map = checkpoint.get("best_map", 0.0)
        self._history = checkpoint.get("history", self._history)

        if self._scheduler is not None and "scheduler_state_dict" in checkpoint:
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self._scaler is not None and "scaler_state_dict" in checkpoint:
            self._scaler.load_state_dict(checkpoint["scaler_state_dict"])

        epoch: int = checkpoint["epoch"]
        self._logger.info(f"チェックポイントを読み込み: epoch={epoch}")

        return epoch

    @property
    def history(self) -> dict[str, list[float]]:
        """学習履歴を取得.

        Returns:
            学習履歴の辞書.
        """
        return self._history

    @property
    def best_map(self) -> float:
        """ベストmAPを取得.

        Returns:
            ベストmAP.
        """
        return self._best_map
