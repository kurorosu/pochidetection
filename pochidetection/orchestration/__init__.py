"""CLI batch 推論フローのオーケストレーション.

pipeline 構築 (``pipelines/``) が返す ``ResolvedPipeline`` を入口に,
画像ループ / レポート出力 / サマリーログを束ねる.
"""

from pochidetection.orchestration.batch_inference import run_batch_inference

__all__ = ["run_batch_inference"]
