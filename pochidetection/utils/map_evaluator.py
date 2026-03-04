"""COCO アノテーションと推論結果から mAP を計算する評価器."""

import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import torch
from torchmetrics.detection import MeanAveragePrecision

from pochidetection.logging import LoggerManager
from pochidetection.scripts.rtdetr.inference.detection import Detection
from pochidetection.utils.benchmark import DetectionMetrics

logger = LoggerManager().get_logger(__name__)


class MapEvaluator:
    """COCO アノテーションと推論結果から mAP を計算する.

    Attributes:
        _annotations: COCO アノテーション辞書.
        _image_id_by_filename: ファイル名から image_id へのマッピング.
        _filenames_by_image_id: image_id からファイル名リストへの逆引きマッピング.
        _gt_by_image_id: image_id ごとのアノテーションリスト.
    """

    _BACKGROUND_NAMES = {"_background_", "background"}

    def __init__(self, annotation_path: Path) -> None:
        """初期化.

        Args:
            annotation_path: COCO フォーマットのアノテーション JSON パス.
        """
        with open(annotation_path, encoding="utf-8") as f:
            self._annotations: dict = json.load(f)

        self._image_id_by_filename: dict[str, int] = {}
        self._filenames_by_image_id: dict[int, list[str]] = {}
        for img in self._annotations["images"]:
            file_name = img["file_name"]
            image_id = img["id"]
            self._image_id_by_filename[file_name] = image_id
            filenames_for_id = [file_name]
            # 学習時は image_id でアノテーションを引くためファイル名マッチが不要だが,
            # 推論時はファイルシステムから画像を列挙するためベースネームでの
            # マッチングが必要. アノテーションの file_name がサブディレクトリ付き
            # (例: "JPEGImages/img.jpg") の場合にも対応する.
            basename = self._extract_basename(file_name)
            if basename != file_name:
                if basename in self._image_id_by_filename:
                    logger.warning(
                        f"basename '{basename}' が重複しています "
                        f"('{file_name}' と既存エントリ). "
                        f"フルパスでのマッチングのみ使用します."
                    )
                else:
                    self._image_id_by_filename[basename] = image_id
                    filenames_for_id.append(basename)
            self._filenames_by_image_id[image_id] = filenames_for_id

        # CocoDetectionDataset と同じリマップ: 背景除外 → カテゴリID昇順ソート → 連続インデックス
        categories = sorted(
            [
                c
                for c in self._annotations.get("categories", [])
                if c["name"].lower() not in self._BACKGROUND_NAMES
            ],
            key=lambda c: c["id"],
        )
        self._category_id_to_idx: dict[int, int] = {
            cat["id"]: idx for idx, cat in enumerate(categories)
        }

        self._gt_by_image_id: dict[int, list[dict]] = {}
        for ann in self._annotations["annotations"]:
            if ann["category_id"] not in self._category_id_to_idx:
                continue
            image_id = ann["image_id"]
            if image_id not in self._gt_by_image_id:
                self._gt_by_image_id[image_id] = []
            self._gt_by_image_id[image_id].append(ann)

    def evaluate(self, predictions: dict[str, list[Detection]]) -> DetectionMetrics:
        """推論結果と GT から mAP を計算する.

        学習時 (train.py) の mAP 計算との違い:
        - 座標系: 学習時は正規化座標, 本クラスはピクセル座標.
          pred/GT が同一スケールのため mAP 値への影響はない.
        - threshold: 学習時は 0.2 固定, 推論時は CLI 指定値 (デフォルト 0.5).
          閾値が高いほど低スコア検出が除外され, recall が下がりうる.
        - NMS: 学習時は未適用, 推論時は適用済み.
          重複除去後の予測で評価するため, 学習時と値が異なりうる.
        - バッチ更新: 学習時は 1画像ずつ update, 推論時は全画像まとめて update.
          torchmetrics の仕様上, 結果は同一.

        Note:
            本メソッドは GT の全画像を起点に走査するため,
            「GT の画像セット = 推論フォルダの画像セット」という前提を必要としない.
            predictions に含まれない GT 画像は空の予測 (検出 0 件) として扱われ,
            その GT オブジェクトは False Negative にカウントされる.

        Args:
            predictions: ファイル名をキー, 検出結果リストを値とする辞書.

        Returns:
            mAP@0.5 と mAP@0.5:0.95 を含む DetectionMetrics.
        """
        metric = MeanAveragePrecision(iou_type="bbox")

        preds_list: list[dict[str, torch.Tensor]] = []
        targets_list: list[dict[str, torch.Tensor]] = []

        # GT の全画像を走査し, predictions にない画像は空の予測として扱う.
        # これにより, 推論対象外の GT が False Negative として正しくカウントされる.
        for image_id, filenames in self._filenames_by_image_id.items():
            # predictions からファイル名で検出結果を検索
            detections: list[Detection] = []
            for fn in filenames:
                if fn in predictions:
                    detections = predictions[fn]
                    break

            # 予測
            if detections:
                boxes = torch.tensor([d.box for d in detections], dtype=torch.float32)
                scores = torch.tensor(
                    [d.score for d in detections], dtype=torch.float32
                )
                labels = torch.tensor([d.label for d in detections], dtype=torch.int64)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                scores = torch.zeros((0,), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            preds_list.append({"boxes": boxes, "scores": scores, "labels": labels})

            # GT
            gt_anns = self._gt_by_image_id.get(image_id, [])
            if gt_anns:
                gt_boxes = torch.tensor(
                    [self._xywh_to_xyxy(ann["bbox"]) for ann in gt_anns],
                    dtype=torch.float32,
                )
                gt_labels = torch.tensor(
                    [self._category_id_to_idx[ann["category_id"]] for ann in gt_anns],
                    dtype=torch.int64,
                )
            else:
                gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
                gt_labels = torch.zeros((0,), dtype=torch.int64)

            targets_list.append({"boxes": gt_boxes, "labels": gt_labels})

        metric.update(preds_list, targets_list)
        result = metric.compute()

        return DetectionMetrics(
            map_50=result["map_50"].item(),
            map_50_95=result["map"].item(),
        )

    @staticmethod
    def _extract_basename(file_name: str) -> str:
        r"""パス区切り文字を考慮してベースネームを抽出する.

        COCO アノテーションの file_name は OS に依存して "/" や "\\" を含む場合がある.
        両方の区切り文字を考慮してベースネームを返す.

        Args:
            file_name: アノテーション内の file_name 文字列.

        Returns:
            ベースネーム (ファイル名のみ).
        """
        # バックスラッシュを含む場合は Windows パスとして解釈
        if "\\" in file_name:
            return PureWindowsPath(file_name).name
        return PurePosixPath(file_name).name

    @staticmethod
    def _xywh_to_xyxy(bbox: list[float]) -> list[float]:
        """COCO の [x, y, w, h] を [x1, y1, x2, y2] に変換する.

        Args:
            bbox: COCO フォーマットの [x, y, w, h].

        Returns:
            [x1, y1, x2, y2] フォーマット.
        """
        x, y, w, h = bbox
        return [x, y, x + w, y + h]
