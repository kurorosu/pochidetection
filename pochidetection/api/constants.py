"""API 全体で共有する定数.

``schemas`` / ``serializers`` の双方で利用する上限値や許容 dtype を集約する.
モジュール間の import 方向を一方向 (``constants`` → ``schemas`` / ``serializers``) に
保ち, 循環 import を避ける目的で切り出している.

Attributes:
    MAX_PIXELS: 受け付ける画像の最大ピクセル数 (height * width).
        4096 x 4096 を上限とし, それを超える raw 配列は検証段階で弾く.
    _ALLOWED_DTYPES: raw フォーマットで許容する numpy dtype 名の集合.
        現状は ``uint8`` のみをサポートする.
"""

MAX_PIXELS: int = 4096 * 4096
_ALLOWED_DTYPES: frozenset[str] = frozenset({"uint8"})
