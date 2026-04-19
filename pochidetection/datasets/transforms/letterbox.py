"""Letterbox を torchvision v2 Transform として提供する."""

from typing import Any

from torchvision.transforms import v2
from torchvision.transforms.v2._utils import query_size

from pochidetection.core.letterbox import apply_letterbox, compute_letterbox_params


class LetterboxTransform(v2.Transform):
    """アスペクト比維持 + padding で target サイズにリサイズする v2 Transform.

    ``v2.Transform`` 規約に従い, 画像 (``PIL.Image`` / ``tv_tensors.Image``) と
    ``tv_tensors.BoundingBoxes`` を同期変換する. 幾何計算は ``core.letterbox`` の
    純粋関数に委譲し, 推論側 (Issue #445) からも同じパラメータ計算 + pixel 操作を
    再利用できる構造とする.

    Attributes:
        size: target サイズ ``(height, width)``.
        pad_value: padding に使うピクセル値 (default ``0``).
    """

    def __init__(
        self,
        size: int | tuple[int, int] | list[int],
        pad_value: int = 0,
    ) -> None:
        """初期化.

        Args:
            size: target サイズ. ``int`` の場合は正方 (size, size), タプル / list の
                場合は ``(height, width)`` として解釈する.
            pad_value: padding に使うピクセル値.

        Raises:
            ValueError: size が 2 要素でない, または非正値を含む場合.
        """
        super().__init__()
        if isinstance(size, int):
            size_hw = (size, size)
        else:
            if len(size) != 2:
                raise ValueError(
                    f"size は int もしくは長さ 2 の (height, width) である必要があります: {size}"
                )
            size_hw = (int(size[0]), int(size[1]))
        if size_hw[0] <= 0 or size_hw[1] <= 0:
            raise ValueError(f"size は正の整数である必要があります: {size_hw}")

        self.size: tuple[int, int] = size_hw
        self.pad_value = pad_value

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Flat inputs から src size を抽出し, letterbox パラメータを計算する.

        Args:
            flat_inputs: v2.Transform が内部で pytree 分解した入力のフラットリスト.

        Returns:
            ``{"params": LetterboxParams}`` を持つ dict.
        """
        src_h, src_w = query_size(flat_inputs)
        params = compute_letterbox_params((src_h, src_w), self.size)
        return {"params": params}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """各 input に letterbox を適用する.

        ``apply_letterbox`` は内部で ``v2.functional.resize`` + ``v2.functional.pad``
        を呼ぶ. これらは tv_tensors を型ディスパッチで扱うため,
        ``PIL.Image`` / ``torch.Tensor`` / ``tv_tensors.Image`` / ``tv_tensors.BoundingBoxes``
        のいずれが入力されても同一パラメータで同期変換される.

        Args:
            inpt: 個別の input (画像 / bbox / mask 等).
            params: ``make_params`` の戻り値.

        Returns:
            letterbox 適用後の input (入力と同じ型).
        """
        return apply_letterbox(inpt, params["params"], pad_value=self.pad_value)
