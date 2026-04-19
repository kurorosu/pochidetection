"""フェーズ別タイミングログの整形ヘルパー.

`POST /api/v1/detect` の INFO ログや他の phase_times dict を扱うログ出力で
共通利用する. GPU / wall clock のフォールバック表示や値欠落時の "N/A" 整形を
1 箇所に集約する.
"""


def format_inference(phase_times: dict[str, float]) -> str:
    """推論フェーズのサマリートークンを整形する.

    GPU 実時間 (``pipeline_inference_gpu_ms``) と wall clock
    (``pipeline_inference_ms``) の双方が揃う場合は ``inf(gpu/wall)=X.X/Y.Y`` 形式,
    wall のみなら ``inf(wall)=Y.Y`` 形式, いずれも欠けていれば ``inf=N/A``.

    Args:
        phase_times: ``pipeline_inference_ms`` / ``pipeline_inference_gpu_ms``
            キーを含みうる ms 値の辞書.

    Returns:
        1 行ログに差し込める形式の文字列.
    """
    inf_wall = phase_times.get("pipeline_inference_ms")
    inf_gpu = phase_times.get("pipeline_inference_gpu_ms")
    if inf_gpu is not None and inf_wall is not None:
        return f"inf(gpu/wall)={inf_gpu:.1f}/{inf_wall:.1f}"
    if inf_wall is not None:
        return f"inf(wall)={inf_wall:.1f}"
    return "inf=N/A"


def format_phase(phase_times: dict[str, float], key: str, label: str) -> str:
    """単一フェーズの ms 値を整形する.

    Args:
        phase_times: ms 値の辞書.
        key: ``phase_times`` から参照するキー (例: ``pipeline_preprocess_ms``).
        label: ログに表示するラベル (例: ``pre``).

    Returns:
        ``<label>=<value:.1f>`` 形式. 値欠落時は ``<label>=N/A``.
    """
    value = phase_times.get(key)
    return f"{label}={value:.1f}" if value is not None else f"{label}=N/A"
