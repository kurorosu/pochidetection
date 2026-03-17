"""COCO データセットのクラス定義.

プリトレインモデル (RT-DETR) 使用時に class_names / num_classes を
自動設定するために使用する.
HuggingFace RT-DETR は連番 0-79 にリマップ済みのため 80 要素リストで対応する.
"""

from pochidetection.configs.schemas import DetectionConfigDict

COCO_CLASS_NAMES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
"""COCO 80 クラス名リスト (連番 0-79)."""

COCO_NUM_CLASSES: int = len(COCO_CLASS_NAMES)
"""COCO クラス数 (80)."""


def build_pretrained_config() -> DetectionConfigDict:
    """プリトレイン推論用の最小 config を生成する.

    ユーザ config を無視し, RT-DETR COCO プリトレインモデルに必要な
    推論パラメータのみで構成する.

    Returns:
        プリトレイン推論用の設定辞書.
    """
    return DetectionConfigDict(
        architecture="RTDetr",
        model_name="PekingU/rtdetr_r50vd",
        num_classes=COCO_NUM_CLASSES,
        class_names=COCO_CLASS_NAMES,
        image_size={"height": 640, "width": 640},
        device="cuda",
        use_fp16=False,
        cudnn_benchmark=False,
        infer_score_threshold=0.5,
        nms_iou_threshold=0.5,
        work_dir="work_dirs",
    )
