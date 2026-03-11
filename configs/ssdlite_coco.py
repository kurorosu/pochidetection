"""SSDLite MobileNetV3 + COCO形式データセット用設定ファイル.

使用方法:
    from pochidetection.utils import ConfigLoader
    config = ConfigLoader.load("configs/ssdlite_coco.py")
"""

# モデル設定
architecture = "SSDLite"
num_classes = 4

# クラス名設定 (推論時の可視化に使用)
class_names = ["pochi", "pochi2", "pochi3", "pochi4"]

# 画像サイズ設定
image_size = {"height": 320, "width": 320}

# データ設定
data_root = "data"
train_split = "train"
val_split = "val"
batch_size = 32

# 学習設定
epochs = 100
learning_rate = 1e-3
lr_scheduler = "CosineAnnealingLR"  # None で無効 (デフォルト)
lr_scheduler_params = {"eta_min": 1e-5}  # Scheduler 固有パラメータ

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
nms_iou_threshold = 0.55  # NMS IoU 閾値 (torchvision の nms_thresh に渡される)

# 推論ベンチマーク設定　指定しないと精度評価無し推論
annotation_path = "data/val/annotations.json"  # mAP 評価用 COCO アノテーション
# 推論対象の画像フォルダ (CLI -d 未指定時に使用)
infer_image_dir = "data/val/JPEGImages"

# ワークスペース設定
work_dir = "work_dirs"
