"""推論時 preprocess 画像のデバッグ保存ユーティリティ.

推論 4 経路 (CLI batch / 動画 / カメラ / WebAPI) で, モデルが実際に見た
**letterbox 適用後の画像** を先頭 N 枚保存する. 学習側 ``debug_save_count`` と
対になる機能で, preprocess (letterbox + resize) に silent bug が無いかを目視
検証する目的. bbox 描画は推論経路側で別途行われるため本ヘルパーでは行わない.

- ``letterbox=True``: ``apply_letterbox`` の結果 (target_hw サイズ, padding あり)
  を保存. モデル入力と完全に同じ形状 / 値域.
- ``letterbox=False``: source image をそのまま保存 (単純 resize 経路の preprocess
  後画像は現状スコープ外).
"""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.core.letterbox import apply_letterbox, compute_letterbox_params

__all__ = ["InferDebugConfig", "save_infer_debug_image"]


@dataclass(frozen=True, slots=True)
class InferDebugConfig:
    """推論 debug 保存の設定バンドル.

    ``process_frames`` など引数が多い loop 関数に渡す際に, 関連する 4 値を
    1 つの immutable 値として束ねる.

    Attributes:
        save_count: 保存する最大枚数. 0 の場合は ``InferDebugConfig`` を
            渡さずに ``None`` にする運用で, 本値が 0 で直接渡されるケースは
            呼出し側の分岐漏れを意味する.
        output_dir: 保存先ディレクトリ. ``infer_XXXX.jpg`` を直下に置く.
        target_hw: letterbox 先 (height, width). ``letterbox=False`` 時は
            未使用.
        letterbox: ``True`` で ``apply_letterbox`` 後画像を保存.
    """

    save_count: int
    output_dir: Path
    target_hw: tuple[int, int] | None
    letterbox: bool

    @classmethod
    def from_config(
        cls,
        config: DetectionConfigDict,
        output_dir: Path,
    ) -> "InferDebugConfig | None":
        """設定辞書と保存先から ``InferDebugConfig`` を組み立てる.

        Args:
            config: 検証済み設定辞書. ``infer_debug_save_count`` /
                ``image_size`` / ``letterbox`` を参照する.
            output_dir: ``infer_debug/`` を置く親ディレクトリ. CLI では
                ``ctx.saver.output_dir`` (model 駆動の
                ``work_dirs/<train>/best/inference_NNN``) を渡す.

        Returns:
            ``infer_debug_save_count > 0`` の場合のみ設定オブジェクトを返し,
            それ以外は ``None``.
        """
        save_count = config.get("infer_debug_save_count", 0)
        if save_count <= 0:
            return None

        image_size = config.get("image_size")
        target_hw = (
            (int(image_size["height"]), int(image_size["width"]))
            if image_size is not None
            else None
        )
        return cls(
            save_count=save_count,
            output_dir=output_dir / "infer_debug",
            target_hw=target_hw,
            letterbox=config.get("letterbox", True),
        )


def save_infer_debug_image(
    source_image: Image.Image,
    target_hw: tuple[int, int] | None,
    letterbox: bool,
    save_path: Path,
) -> None:
    """推論時 preprocess 後画像を JPEG で保存する.

    Args:
        source_image: 元画像 (PIL RGB).
        target_hw: letterbox 先 (height, width). ``letterbox=False`` や ``None``
            の場合は無視される.
        letterbox: ``True`` で ``apply_letterbox`` の結果を保存, ``False`` で
            source をそのまま保存.
        save_path: 保存先ファイルパス (``infer_XXXX.jpg`` 等). 親ディレクトリは
            本関数が ``mkdir(parents=True, exist_ok=True)`` で作成する.
    """
    if letterbox and target_hw is not None:
        src_w, src_h = source_image.size
        params = compute_letterbox_params((src_h, src_w), target_hw)
        debug_image = apply_letterbox(source_image, params, pad_value=0)
    else:
        debug_image = source_image

    save_path.parent.mkdir(parents=True, exist_ok=True)
    debug_image.save(save_path, format="JPEG")
