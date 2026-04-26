"""pochi prepare-demo サブコマンド.

RT-DETR COCO プリトレインから ONNX エクスポート → TensorRT FP16 engine ビルド
までを 1 コマンドで実行し, ``pochi serve -m work_dirs/<run>/best/model_fp16.engine``
で WebAPI を即起動できる成果物を生成する.

[#441](https://github.com/kurorosu/pochidetection/issues/441) (zero-arg
``pochi serve`` プリトレイン起動) は補完関係: 非 TRT 環境向け. 本コマンドは
TensorRT 必須.
"""

import argparse
import sys

from pochidetection.configs.schemas import DetectionConfigDict, ImageSizeDict
from pochidetection.core.coco_classes import COCO_CLASS_NAMES, COCO_NUM_CLASSES
from pochidetection.logging import LoggerManager
from pochidetection.models.rtdetr import RTDetrModel
from pochidetection.scripts.rtdetr.export_onnx import export_onnx
from pochidetection.utils.work_dir import WorkspaceManager

logger = LoggerManager().get_logger(__name__)

DEMO_MODEL_NAME: str = "PekingU/rtdetr_r50vd"
"""RT-DETR COCO プリトレインの HF モデル ID. デモ用固定."""


def _check_tensorrt_available() -> None:
    """インストール状態を確認し, TensorRT が無ければ早期に exit する.

    workspace 作成 / HF DL より前に呼ぶことで副作用ゼロを保証する.

    Raises:
        SystemExit: TensorRT が未インストールの場合.
    """
    try:
        import pochidetection.tensorrt  # noqa: F401
    except ImportError:
        logger.error(
            "TensorRT がインストールされていません. 本コマンドは TensorRT 必須です. "
            "非 TRT 環境では `pochi serve` を引数なしで起動してください (#441)."
        )
        sys.exit(1)


def build_demo_config(image_size: ImageSizeDict) -> DetectionConfigDict:
    """RT-DETR COCO プリトレイン用の最小 config を組み立てる.

    ``DetectionConfig`` (Pydantic, ``extra="forbid"``) を通る最小キーセット.
    ``model_name`` / ``architecture`` は schema default (RTDetr / r50vd) と一致.

    Args:
        image_size: 入力画像サイズ ``{"height": int, "width": int}``.

    Returns:
        ``ConfigLoader.write_config`` で書き出し可能な dict.
    """
    return {
        "architecture": "RTDetr",
        "model_name": DEMO_MODEL_NAME,
        "num_classes": COCO_NUM_CLASSES,
        "class_names": list(COCO_CLASS_NAMES),
        "image_size": image_size,
        "data_root": ".",
        "device": "cuda",
        "use_fp16": False,
        "infer_score_threshold": 0.5,
        "nms_iou_threshold": 0.5,
        "letterbox": True,
    }


def run_prepare_demo(args: argparse.Namespace) -> None:
    """``pochi prepare-demo`` の dispatcher.

    1. TRT 利用可能性を早期検出
    2. ``WorkspaceManager`` で run ディレクトリ作成
    3. RT-DETR COCO プリトレインを HF からダウンロードし ``best/`` に保存
    4. ``<run>/config.py`` を自動生成
    5. ONNX エクスポート → ``best/model.onnx``
    6. TensorRT FP16 engine ビルド → ``best/model_fp16.engine``
    7. ``pochi serve`` 起動コマンドを案内

    Args:
        args: パース済みの引数 (``args.input_size`` を参照).
    """
    logger.info(
        "=== pochi prepare-demo: RT-DETR COCO プリトレイン → TRT FP16 engine ==="
    )

    _check_tensorrt_available()

    height, width = args.input_size
    image_size: ImageSizeDict = {"height": height, "width": width}

    workspace = WorkspaceManager()
    run_dir = workspace.create_workspace()
    best_dir = workspace.get_best_dir()
    best_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created workspace: {run_dir}")

    logger.info(f"Downloading RT-DETR COCO pretrained ({DEMO_MODEL_NAME})...")
    model = RTDetrModel(
        model_name=DEMO_MODEL_NAME,
        pretrained=True,
        image_size=image_size,
    )
    model.save(best_dir)
    logger.info(f"Saved pretrained weights to {best_dir}")

    config = build_demo_config(image_size)
    config_path = workspace.save_config(config, "config.py")
    logger.info(f"Generated {config_path}")

    logger.info("Exporting ONNX...")
    export_onnx(
        config=config,
        model_dir=str(best_dir),
        output=None,
        opset_version=17,
        input_size=(height, width),
        skip_verify=False,
    )

    onnx_path = best_dir / "model.onnx"
    engine_path = best_dir / "model_fp16.engine"
    logger.info("Building TensorRT FP16 engine...")
    # TRT availability は冒頭で確認済み.
    from pochidetection.tensorrt import TensorRTExporter

    TensorRTExporter().export(
        onnx_path=onnx_path,
        output_path=engine_path,
        input_size=(height, width),
        use_fp16=True,
    )

    logger.info("=" * 60)
    logger.info("Done. To start the WebAPI server:")
    logger.info(f"  uv run pochi serve -m {engine_path}")
    logger.info("=" * 60)
