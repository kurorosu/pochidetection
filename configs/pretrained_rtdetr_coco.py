"""RT-DETR COCO プリトレインモデル用設定ファイル.

モデルパス未指定時のフォールバックとして使用する.
推論パラメータのみで構成し, 学習・データセット関連の設定は含まない.

初回実行時に HuggingFace Hub からモデルをダウンロードする.
ダウンロード完了後は local_files_only = True に変更することで,
ネットワークアクセスなしで推論可能になる.
"""

from pochidetection.scripts.common.coco_classes import COCO_CLASS_NAMES

# モデル設定
architecture = "RTDetr"
model_name = "PekingU/rtdetr_r18vd"
local_files_only = False  # True にするとキャッシュ済みモデルのみ使用 (オフライン対応)

# クラス設定
num_classes = 80
class_names = COCO_CLASS_NAMES

# 画像サイズ設定
image_size = {"height": 640, "width": 640}

# デバイス設定
device = "cuda"
cudnn_benchmark = False
use_fp16 = False

# Threshold
infer_score_threshold = 0.5
nms_iou_threshold = 0.5

# カメラ設定 (Webcam リアルタイム推論用)
camera_fps = 60  # カメラ FPS (None でカメラのデフォルトを使用)
camera_resolution = [640, 480]  # カメラ解像度 [width, height] (None でデフォルト)

# ワークスペース設定
work_dir = "work_dirs"

# 学習設定 (プリトレイン推論では使用しないが, スキーマバリデーション用)
data_root = "."
