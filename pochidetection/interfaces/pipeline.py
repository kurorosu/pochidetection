"""推論パイプラインの抽象インターフェース."""

from abc import ABC, abstractmethod

from PIL import Image

from pochidetection.core.detection import Detection
from pochidetection.utils import PhasedTimer


class IDetectionPipeline(ABC):
    """推論パイプラインの共通インターフェース.

    前処理・推論・後処理の 3 フェーズで構成される E2E 推論パイプラインの
    共通契約を定義する. 各フェーズの具体的なシグネチャは実装に依存するため,
    本インターフェースでは E2E 実行メソッド ``run`` と
    フェーズ別タイマー ``phased_timer`` のみを規定する.
    """

    PHASES = ["preprocess", "inference", "postprocess"]

    @abstractmethod
    def run(self, image: Image.Image) -> list[Detection]:
        """E2E 推論を実行する.

        Args:
            image: 入力画像 (PIL Image).

        Returns:
            検出結果のリスト.
        """

    @property
    @abstractmethod
    def phased_timer(self) -> PhasedTimer | None:
        """フェーズ別タイマーを取得."""
