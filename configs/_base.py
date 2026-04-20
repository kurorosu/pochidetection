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
epochs = 20
lr_scheduler = "CosineAnnealingLR"  # None で無効 (デフォルト)

# Early Stopping 設定
early_stopping_patience = 20  # 改善なしで停止するまでのエポック数 (None or 0 で無効)
early_stopping_metric = "mAP"  # "mAP" (高い方が良い) or "val_loss" (低い方が良い)
early_stopping_min_delta = 0.001  # 改善と見なす最小変化量

# デバイス設定
device = "cuda"
cudnn_benchmark = False  # 入力サイズ固定時に学習・推論を高速化
use_fp16 = False  # FP16 推論 (CUDA のみ)
enable_tensorboard = True  # TensorBoard によるメトリクス記録

# Threshold
train_score_threshold = 0.2  # 学習時の mAP 計算で使用するスコア閾値
infer_score_threshold = 0.2  # 推論時の検出信頼度閾値 (この値未満の検出を除外)

# 推論ベンチマーク設定　指定しないと精度評価無し推論
annotation_path = "data/val/annotations.json"  # mAP 評価用 COCO アノテーション
infer_image_dir = "data/val/JPEGImages"  # 推論対象の画像フォルダ

# デバッグ画像保存 (augmentation の有無に関わらず, 1 エポック目の先頭 N 枚を保存)
# 保存先: ``work_dirs/YYYYMMDD_XXX/train_debug/train_XXXX.jpg``
# letterbox / preprocess の silent bug を目視検知する目的. 0 で無効.
debug_save_count = 10

# 推論時 preprocess 後画像のデバッグ保存 (letterbox 適用後, bbox なし)
# 保存先: CLI は ``{output_dir}/infer_debug/``, WebAPI は
# ``work_dirs/api_<timestamp>/infer_debug/``. 0 で無効.
infer_debug_save_count = 1

# Letterbox (アスペクト比維持 + padding) リサイズ
# 学習 / 推論 両方に適用. False で従来の単純 resize に戻る.
letterbox = True

# Data Augmentation 設定 (学習時のみ適用, 詳細は docs/augmentation.md を参照)
augmentation = {
    "enabled": False,  # データ拡張の有効化 (True で transforms の変換を適用)
    "transforms": [
        # 幾何変換 (bbox も自動変換)
        {"name": "RandomHorizontalFlip", "p": 0.5},
        {"name": "RandomVerticalFlip", "p": 0.1},
        # {"name": "RandomRotation", "degrees": 10},
        # {"name": "RandomAffine", "degrees": 0,
        #   "translate": [0.1, 0.1], "scale": [0.9, 1.1]},
        # {"name": "RandomPerspective", "distortion_scale": 0.1, "p": 0.2},
        {"name": "RandomZoomOut", "p": 0.2, "side_range": [1.0, 1.5]},
        # 色変換 (対象の色が特徴の場合はコメントアウトを推奨)
        # {"name": "ColorJitter", "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
    ],
}

# カメラ設定 (Webcam リアルタイム推論用)
camera_fps = 60.0  # カメラ FPS (None でカメラのデフォルトを使用)
camera_resolution = [640, 480]  # カメラ解像度 [width, height] (None でデフォルト)

# ワークスペース設定
work_dir = "work_dirs"
