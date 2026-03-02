"""RT-DETR + COCO形式データセット用設定ファイル.

使用方法:
    from pochidetection.utils import ConfigLoader
    config = ConfigLoader.load("configs/rtdetr_coco.py")
"""

# モデル設定
model_name = "PekingU/rtdetr_r50vd"
num_classes = 4

# クラス名設定 (推論時の可視化に使用)
class_names = ["pochi", "pochi2", "pochi3", "pochi4"]

# 画像サイズ設定
image_size = {"height": 640, "width": 640}

# データ設定
data_root = "data"
train_split = "train"
val_split = "val"
batch_size = 8

# 学習設定
epochs = 5
learning_rate = 1e-4

# デバイス設定
device = "cuda"
cudnn_benchmark = False  # 入力サイズ固定時に推論高速化
use_fp16 = False  # FP16 推論 (CUDA のみ)

# Threshold
train_score_threshold = 0.2
infer_score_threshold = 0.5
nms_iou_threshold = 0.5

# 推論ベンチマーク設定
annotation_path = "data/val/annotations.json"  # mAP 評価用 COCO アノテーション

# ワークスペース設定
work_dir = "work_dirs"
