"""``POST /detect`` の bbox が元画像座標系で返ることを検証.

Issue #442 の意図 (PR #440 レビュー対応): 推論パイプラインで letterbox
有効時に, ``/detect`` レスポンスの bbox が「letterbox 後 (target_hw=640x640)
空間」ではなく「元画像 (例: 1280x720)」空間の値であることを保証する.
逆変換 ``(box - pad) / scale`` が落ちると bbox は target_hw に閉じ込められ,
クライアント側で誤った位置に矩形を描いてしまう.

検出される失敗モード: 1280x720 (target_hw 両軸より大) で letterbox 逆変換が
落ちると bbox は target 空間 (max 640) に閉じ込められ, max(x2) <= 640 かつ
max(y2) <= 640 になる. 逆変換が効いていれば 1280x720 全域に bbox が広がるため
max(x2) は target_w を超える ((640 - pad_left) / scale で 1280 まで).
"""

import base64
from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pochidetection.api.app import build_engine, create_app
from pochidetection.api.backends import IDetectionBackend
from pochidetection.api.config import ServerConfig
from pochidetection.api.state import set_engine

_RESOLUTIONS = [(640, 480), (1280, 720)]
_LARGE_RESOLUTION = (1280, 720)


def _make_synthetic_bgr_image(width: int, height: int) -> np.ndarray:
    """検出が複数発生しやすい決定的 BGR uint8 画像を生成する."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = (180, 50, 50)
    image[:, width // 2 :] = (50, 180, 50)
    cv2.rectangle(
        image,
        (width // 4, height // 4),
        (3 * width // 4, 3 * height // 4),
        (50, 50, 200),
        thickness=-1,
    )
    return image


def _post_detect(
    client: TestClient, bgr: np.ndarray, score_threshold: float
) -> list[dict]:
    """``POST /detect`` を raw 形式で実行し, ``detections`` を返す."""
    height, width = bgr.shape[:2]
    payload = {
        "image_data": base64.b64encode(bgr.tobytes()).decode("ascii"),
        "format": "raw",
        "shape": [height, width, 3],
        "dtype": "uint8",
        "score_threshold": score_threshold,
    }
    res = client.post("/api/v1/detect", json=payload)
    assert res.status_code == 200, res.text
    body = res.json()
    return list(body["detections"])


@pytest.mark.slow
class TestDetectBboxInOriginalImageSpace:
    """``/detect`` レスポンスの bbox が元画像座標系であることを検証.

    backend は class スコープで一度だけ構築する (RT-DETR COCO プリトレインの
    HF cache miss で初回 ~50MB DL が発生するため).
    """

    @pytest.fixture(scope="class")
    def backend(self) -> Iterator[IDetectionBackend]:
        """RT-DETR COCO プリトレイン backend を 1 度だけロード."""
        engine = build_engine(ServerConfig(model_path=None))
        try:
            yield engine
        finally:
            engine.close()

    @pytest.fixture
    def client(self, backend: IDetectionBackend) -> Iterator[TestClient]:
        """TestClient. 各テストで engine を state に注入する."""
        set_engine(backend)
        app = create_app(None)
        with TestClient(app) as test_client:
            yield test_client

    @pytest.mark.parametrize(("width", "height"), _RESOLUTIONS)
    def test_detect_returns_detections_at_resolution(
        self,
        client: TestClient,
        width: int,
        height: int,
    ) -> None:
        """各解像度でプリトレインモデルが検出を返す (pipeline 動作の sanity)."""
        bgr = _make_synthetic_bgr_image(width, height)
        api_dets = _post_detect(client, bgr, score_threshold=0.0)
        assert len(api_dets) > 0

    def test_bbox_max_coord_exceeds_target_hw_for_large_input(
        self,
        backend: IDetectionBackend,
        client: TestClient,
    ) -> None:
        """target_hw より大きい入力で bbox の最大座標が target_hw を超える.

        1280x720 (両軸とも target=640 より大) で letterbox 逆変換が落ちると
        bbox は target 空間 (max 640) に閉じ込められ, max(x2) <= 640 になる.
        逆変換が効いていれば 1280x720 全域に bbox が広がるため max(x2) は
        target_w を超える ((640 - pad_left) / scale で最大 1280 まで).
        """
        target_hw = backend._config["image_size"]
        target_h, target_w = target_hw["height"], target_hw["width"]
        width, height = _LARGE_RESOLUTION
        assert (
            width > target_w and height > target_h
        ), "本テストは入力が target_hw 両軸より大きいことを前提にしている"

        bgr = _make_synthetic_bgr_image(width, height)
        api_dets = _post_detect(client, bgr, score_threshold=0.0)
        assert len(api_dets) > 0

        # 逆変換ありなら max は image_w / (640-pad)/scale=image_h まで届きうる.
        # 逆変換欠落時は max が target_hw + 浮動小数誤差 (~640.1) にしかならないため,
        # target * 1.2 でクリーンに弁別する (実測: 正常系 max=1000-1280, 欠落系 ~640).
        margin = 1.2
        max_x2 = max(det["bbox"][2] for det in api_dets)
        max_y2 = max(det["bbox"][3] for det in api_dets)
        assert max_x2 > target_w * margin, (
            f"max(x2)={max_x2:.1f} <= target_w * {margin} ({target_w * margin:.1f}). "
            f"letterbox 逆変換欠落の可能性 (image={width}x{height})"
        )
        assert max_y2 > target_h * margin, (
            f"max(y2)={max_y2:.1f} <= target_h * {margin} ({target_h * margin:.1f}). "
            f"letterbox 逆変換欠落の可能性 (image={width}x{height})"
        )
