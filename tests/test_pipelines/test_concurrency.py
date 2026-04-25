"""Pipeline のマルチ thread 呼出での request-scoped state 検証.

letterbox 幾何パラメータが pipeline インスタンス属性ではなく preprocess 戻り値で
受け渡される (call stack 上の local 変数) ことで, 同一 pipeline インスタンスを
複数 thread が並行利用しても bbox 逆変換が混線しないことを検証する.

Issue #569 の acceptance criteria:
    > マルチ thread 呼出でも params が混線しないことを classical test で検証
    > (2 並行 thread で異なる shape の画像を渡し, bbox が各画像の座標系に正しく戻るか確認)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from pochidetection.interfaces import IInferenceBackend
from pochidetection.pipelines.rtdetr_pipeline import RTDetrPipeline
from pochidetection.pipelines.ssd_pipeline import SsdPipeline

# 画像サイズは 64x64 固定, pipeline は共通 image_size=(64, 64) で letterbox 有効.
# 異なる aspect ratio の入力画像で params (scale, pad) が変わるため race があれば
# bbox の期待値が崩れる.

RTDETR_TRANSFORM = v2.Compose(
    [
        v2.Resize((64, 64), interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

# HF processor が [[10, 20, 30, 40]] を返した時に letterbox 逆変換後の期待値は
# 画像サイズ (letterbox params) に応じて変わる. 表は Sprint Contract の
# acceptance criteria で言う「画像サイズ別の正解値」の根拠.
RTDETR_CASES: list[tuple[tuple[int, int], list[float]]] = [
    # (image_size(w, h), expected_box_after_inverse_letterbox)
    # 正方 64x64: scale=1, pad=0 → (10, 20, 30, 40)
    ((64, 64), [10.0, 20.0, 30.0, 40.0]),
    # 横長 128x32: scale=0.5, pad_top=24, pad_left=0
    # → ((10-0)/0.5, (20-24)/0.5, (30-0)/0.5, (40-24)/0.5) = (20, -8, 60, 32)
    ((128, 32), [20.0, -8.0, 60.0, 32.0]),
    # 縦長 32x128: scale=0.5, pad_top=0, pad_left=24
    # → ((10-24)/0.5, (20-0)/0.5, (30-24)/0.5, (40-0)/0.5) = (-28, 40, 12, 80)
    ((32, 128), [-28.0, 40.0, 12.0, 80.0]),
]


class _RTDetrDummyBackend(IInferenceBackend[tuple[torch.Tensor, torch.Tensor]]):
    """複数 thread が共有しても安全な dummy backend (stateless)."""

    def infer(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """固定 logits / boxes を返す (パラメータ固有の値は返さない)."""
        return torch.zeros((1, 100, 2)), torch.zeros((1, 100, 4))


class _RTDetrDummyProcessor:
    """固定 box [[10, 20, 30, 40]] を返す dummy processor (stateless)."""

    def post_process_object_detection(
        self, outputs: Any, target_sizes: Any, threshold: float
    ) -> list[dict[str, Any]]:
        """固定 bbox を返す (letterbox 後 pixel 座標を模擬)."""
        return [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            }
        ]


class TestRTDetrPipelineConcurrency:
    """RTDetrPipeline を 3 thread で並行呼出しても letterbox 逆変換が混線しない."""

    def test_letterbox_params_are_request_scoped(self) -> None:
        """異なる shape を 3 thread 並行で run() し, 各 thread が期待する bbox を得る."""
        pipeline = RTDetrPipeline(
            backend=_RTDetrDummyBackend(),
            processor=_RTDetrDummyProcessor(),
            transform=RTDETR_TRANSFORM,
            device="cpu",
            image_size=(64, 64),
            letterbox=True,
        )

        def worker(
            case: tuple[tuple[int, int], list[float]],
        ) -> tuple[tuple[int, int], list[float], list[float]]:
            (w, h), expected = case
            image = Image.new("RGB", (w, h))
            # thread 並行性を実際に発生させるため, 各 thread で複数回 run() する.
            last_box = [0.0, 0.0, 0.0, 0.0]
            for _ in range(20):
                detections = pipeline.run(image)
                assert len(detections) == 1
                last_box = detections[0].box
            return (w, h), expected, last_box

        with ThreadPoolExecutor(max_workers=len(RTDETR_CASES)) as pool:
            futures = [pool.submit(worker, case) for case in RTDETR_CASES]
            results = [f.result() for f in as_completed(futures)]

        for (w, h), expected, actual in results:
            assert actual == pytest.approx(
                expected, abs=1e-3
            ), f"image=({w}, {h}): expected {expected}, got {actual}"


# --------------------------------------------------------------------------
# SSD pipeline
# --------------------------------------------------------------------------


class _SsdDummyBackend(IInferenceBackend[dict[str, torch.Tensor]]):
    """複数 thread で共有する stateless な dummy backend.

    letterbox 後 pixel 座標 [[50, 40, 100, 80]] を固定で返す.
    """

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """固定 predictions を返す."""
        return {
            "boxes": torch.tensor([[50.0, 40.0, 100.0, 80.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }


# image_size=(320, 320), letterbox=True, backend は [[50, 40, 100, 80]] 固定.
# 期待値は画像サイズに応じて scale / pad が変わった結果.
SSD_CASES: list[tuple[tuple[int, int], list[float]]] = [
    # 正方 320x320: scale=1, pad=0 → (50, 40, 100, 80)
    ((320, 320), [50.0, 40.0, 100.0, 80.0]),
    # 横長 640x480: scale=0.5, pad_top=40, pad_left=0
    # → ((50-0)/0.5, (40-40)/0.5, (100-0)/0.5, (80-40)/0.5) = (100, 0, 200, 80)
    ((640, 480), [100.0, 0.0, 200.0, 80.0]),
    # 縦長 480x640: scale=0.5, pad_top=0, pad_left=40
    # → ((50-40)/0.5, (40-0)/0.5, (100-40)/0.5, (80-0)/0.5) = (20, 80, 120, 160)
    ((480, 640), [20.0, 80.0, 120.0, 160.0]),
]

SSD_TRANSFORM = v2.Compose(
    [
        v2.Resize((320, 320)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


class TestSsdPipelineConcurrency:
    """SsdPipeline を 3 thread で並行呼出しても letterbox 逆変換が混線しない."""

    def test_letterbox_params_are_request_scoped(self) -> None:
        """異なる shape を 3 thread 並行で run() し, 各 thread が期待する bbox を得る."""
        pipeline = SsdPipeline(
            backend=_SsdDummyBackend(),
            transform=SSD_TRANSFORM,
            image_size=(320, 320),
            device="cpu",
            threshold=0.0,
            letterbox=True,
        )

        def worker(
            case: tuple[tuple[int, int], list[float]],
        ) -> tuple[tuple[int, int], list[float], list[float]]:
            (w, h), expected = case
            image = Image.new("RGB", (w, h))
            last_box = [0.0, 0.0, 0.0, 0.0]
            for _ in range(20):
                detections = pipeline.run(image)
                assert len(detections) == 1
                last_box = detections[0].box
            return (w, h), expected, last_box

        with ThreadPoolExecutor(max_workers=len(SSD_CASES)) as pool:
            futures = [pool.submit(worker, case) for case in SSD_CASES]
            results = [f.result() for f in as_completed(futures)]

        for (w, h), expected, actual in results:
            assert actual == pytest.approx(
                expected, abs=1e-3
            ), f"image=({w}, {h}): expected {expected}, got {actual}"
