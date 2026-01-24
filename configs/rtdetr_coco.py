"""RT-DETR + COCO形式データセット用設定ファイル.

使用方法:
    from pochidetection.utils import ConfigLoader
    config = ConfigLoader.load("configs/rtdetr_coco.py")
"""

# モデル設定
model_name = "PekingU/rtdetr_r50vd"
num_classes = 1

# データ設定
data_root = "data"
train_split = "train"
val_split = "val"
batch_size = 8

# 学習設定
epochs = 50
learning_rate = 1e-3

# デバイス設定
device = "cuda"

# ワークスペース設定
work_dir = "work_dirs"
