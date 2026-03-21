"""推論結果を保存するクラス."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from pochidetection.core.detection import Detection
    from pochidetection.visualization.label_mapper import LabelMapper


class InferenceSaver:
    """推論結果を保存.

    Attributes:
        _output_dir: 出力ディレクトリ.
    """

    def __init__(self, base_dir: Path) -> None:
        """InferenceSaverを初期化.

        inference_001, inference_002 のようにインクリメントしたディレクトリを作成.

        Args:
            base_dir: 親ディレクトリのパス (model_path).
        """
        self._output_dir = self._create_numbered_dir(base_dir)

    def _create_numbered_dir(self, base_dir: Path) -> Path:
        """連番付きディレクトリを作成.

        Args:
            base_dir: 親ディレクトリ.

        Returns:
            作成したディレクトリのパス.
        """
        # 既存の inference_XXX ディレクトリを検索 (任意桁数に対応)
        pattern = re.compile(r"^inference_(\d+)$")
        max_num = 0
        if not base_dir.exists():
            base_dir.mkdir(parents=True)
        for d in base_dir.iterdir():
            if d.is_dir():
                m = pattern.match(d.name)
                if m:
                    max_num = max(max_num, int(m.group(1)))

        next_num = max_num + 1
        output_dir = base_dir / f"inference_{next_num:03d}"
        output_dir.mkdir(parents=True)
        return output_dir

    def save(self, image: Image.Image, filename: str) -> Path:
        """画像を保存.

        Args:
            image: 保存する画像.
            filename: 元のファイル名.

        Returns:
            保存先のパス.
        """
        # ファイル名に _result を付加
        path = Path(filename)
        output_filename = f"{path.stem}_result{path.suffix}"
        output_path = self._output_dir / output_filename

        image.save(output_path)
        return output_path

    def save_crops(
        self,
        image: Image.Image,
        detections: list[Detection],
        filename: str,
        label_mapper: LabelMapper | None = None,
    ) -> list[Path]:
        """検出ボックスのクロップ画像を保存.

        Args:
            image: 元画像.
            detections: 検出結果リスト.
            filename: 元のファイル名.
            label_mapper: クラス ID をラベル名に変換するマッパー.

        Returns:
            保存先パスのリスト.
        """
        if not detections:
            return []

        crop_dir = self._output_dir / "crop"
        crop_dir.mkdir(exist_ok=True)

        stem = Path(filename).stem
        saved: list[Path] = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det.box]
            crop = image.crop((x1, y1, x2, y2))

            if label_mapper is not None:
                label = label_mapper.get_label(det.label)
            else:
                label = str(det.label)

            crop_path = crop_dir / f"{stem}_{i}_{label}_{det.score:.2f}.jpg"
            crop.save(crop_path)
            saved.append(crop_path)

        return saved

    @property
    def output_dir(self) -> Path:
        """出力ディレクトリを取得.

        Returns:
            出力ディレクトリのパス.
        """
        return self._output_dir
