"""推論関連の共有型定義.

循環インポートを回避するため, 複数モジュールで参照される型を
このモジュールに集約する.
"""

from collections.abc import Callable
from pathlib import Path

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.pipelines.context import PipelineContext

BuildPipelineFn = Callable[[DetectionConfigDict, Path], PipelineContext]
"""パイプライン構築コールバックの型.

(config, model_path) を受け取り PipelineContext を返す.
"""
