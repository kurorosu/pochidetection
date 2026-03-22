"""推論関連の共有型定義.

循環インポートを回避するため, 複数モジュールで参照される型を
このモジュールに集約する.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pochidetection.configs.schemas import DetectionConfigDict

SetupPipelineFn = Callable[[DetectionConfigDict, Path], Any]
"""パイプライン構築コールバックの型.

(config, model_path) を受け取り PipelineContext を返す.
戻り値は循環インポート回避のため Any としている.
"""
