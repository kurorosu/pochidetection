"""WebAPI プロセスで共有する検出バックエンドのグローバル状態.

`api/app.py` と `api/routers/*.py` の間で発生していた関数内遅延 import による
循環依存を解消するため, エンジン参照を本モジュールに切り出している.
"""

from pochidetection.api.backends import IDetectionBackend

_engine: IDetectionBackend | None = None


def get_engine() -> IDetectionBackend:
    """Return the globally registered detection backend.

    Returns:
        現在登録されている ``IDetectionBackend`` インスタンス.

    Raises:
        RuntimeError: バックエンドが初期化されていない (``set_engine`` が
            未呼び出し, または ``None`` で解除済み) の場合.
    """
    if _engine is None:
        raise RuntimeError("モデルが初期化されていません")
    return _engine


def set_engine(engine: IDetectionBackend | None) -> None:
    """Register or clear the global detection backend.

    Args:
        engine: 登録する ``IDetectionBackend`` インスタンス. ``None`` を渡すと
            現在のエンジン参照をクリアする (shutdown 用途).
    """
    global _engine
    _engine = engine
