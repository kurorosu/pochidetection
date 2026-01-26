"""RT-DETR + COCO形式データセット用設定ファイル.

使用方法:
    from pochidetection.utils import ConfigLoader
    config = ConfigLoader.load("configs/rtdetr_coco.py")
"""

# モデル設定
model_name = "PekingU/rtdetr_r50vd"
num_classes = 1

# クラス名設定 (推論時の可視化に使用)
class_names = ["pochi"]

# 画像サイズ設定
image_size = {"height": 640, "width": 640}

# データ設定
data_root = "data"
train_split = "train"
val_split = "val"
batch_size = 8

# 学習設定
epochs = 10
learning_rate = 1e-3

# デバイス設定
device = "cuda"
cudnn_benchmark = False  # 入力サイズ固定時に推論高速化
use_fp16 = False  # FP16 推論 (CUDA のみ)

# ワークスペース設定
work_dir = "work_dirs"
