"""SSDLite MobileNetV3 + COCO形式データセット用設定ファイル.

使用方法:
    from pochidetection.utils import ConfigLoader
    config = ConfigLoader.load("configs/ssdlite_coco.py")
"""

# モデル設定
architecture = "SSDLite"

# 画像サイズ設定
image_size = {"height": 320, "width": 320}

# データ設定
batch_size = 8

# 学習設定
learning_rate = 1e-3
lr_scheduler_params = {"eta_min": 1e-3}  # Scheduler 固有パラメータ

# NMS 設定
nms_iou_threshold = 0.5  # NMS IoU 閾値 (torchvision の nms_thresh に渡される)
