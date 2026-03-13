"""全アーキテクチャ共通のベース設定.

アーキテクチャ固有の設定ファイル (rtdetr_coco.py, ssdlite_coco.py) から
インポートして使用する. 必要に応じて各設定ファイルで値を上書きする.
"""

# クラス設定
num_classes = 4
class_names = ["pochi", "pochi2", "pochi3", "pochi4"]

# データ設定
data_root = "data"
train_split = "train"
val_split = "val"

# 学習設定
epochs = 100
lr_scheduler = "CosineAnnealingLR"  # None で無効 (デフォルト)

# Early Stopping 設定
early_stopping_patience = 20  # 改善なしで停止するまでのエポック数 (None or 0 で無効)
early_stopping_metric = "mAP"  # "mAP" (高い方が良い) or "val_loss" (低い方が良い)
early_stopping_min_delta = 0.001  # 改善と見なす最小変化量

# デバイス設定
device = "cuda"
cudnn_benchmark = False  # 入力サイズ固定時に推論高速化
use_fp16 = False  # FP16 推論 (CUDA のみ)

# Threshold
train_score_threshold = 0.2  # 学習時の mAP 計算で使用するスコア閾値
infer_score_threshold = 0.2  # 推論時の検出信頼度閾値 (この値未満の検出を除外)

# 推論ベンチマーク設定　指定しないと精度評価無し推論
annotation_path = "data/val/annotations.json"  # mAP 評価用 COCO アノテーション
infer_image_dir = "data/val/JPEGImages"  # 推論対象の画像フォルダ

# ワークスペース設定
work_dir = "work_dirs"
