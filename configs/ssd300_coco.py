"""SSD300 VGG16 + COCO形式データセット用設定ファイル.

使用方法:
    from pochidetection.utils import ConfigLoader
    config = ConfigLoader.load("configs/ssd300_coco.py")
"""

# モデル設定
architecture = "SSD300"

# 画像サイズ設定
image_size = {"height": 300, "width": 300}

# データ設定
batch_size = 16

# 学習設定
learning_rate = 1e-3
lr_scheduler_params = {"eta_min": 1e-5}  # Scheduler 固有パラメータ
