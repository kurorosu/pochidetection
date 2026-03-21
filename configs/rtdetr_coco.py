"""RT-DETR + COCO形式データセット用設定ファイル.

使用方法:
    from pochidetection.utils import ConfigLoader
    config = ConfigLoader.load("configs/rtdetr_coco.py")
"""

# モデル設定
model_name = "PekingU/rtdetr_r18vd"
local_files_only = False  # True にするとキャッシュ済みモデルのみ使用 (オフライン対応)

# 画像サイズ設定
image_size = {"height": 640, "width": 640}

# データ設定
batch_size = 8

# 学習設定
learning_rate = 1e-4
lr_scheduler_params = {"eta_min": 1e-5}  # Scheduler 固有パラメータ

# NMS 設定
nms_iou_threshold = 0.5  # 推論時の NMS IoU 閾値 (重複検出の除去)
