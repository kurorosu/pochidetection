"""TensorRTエクスポート機能を提供するモジュール."""

import logging
from pathlib import Path

try:
    import tensorrt as trt

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False

from pochidetection.logging import LoggerManager

logger: logging.Logger = LoggerManager().get_logger(__name__)


class TensorRTExporter:
    """ONNXモデルからTensorRTエンジンへの変換を行うクラス.

    FP32 精度のエンジン出力に対応し, Dynamic Batching 用の
    最適化プロファイルを設定してビルドを行う.
    """

    def __init__(self) -> None:
        """初期化.

        Raises:
            ImportError: tensorrt がインストールされていない場合.
        """
        if not _TRT_AVAILABLE:
            raise ImportError(
                "tensorrt パッケージがインストールされていません. "
                "GPU環境構築手順に従って TensorRT をインストールしてください."
            )

        # ログレベルの設定. ERROR のみ TRT 側のログを出力させる.
        self.trt_logger = trt.Logger(trt.Logger.ERROR)

    def export(
        self,
        onnx_path: Path | str,
        output_path: Path | str,
        input_size: tuple[int, int],
        min_batch: int = 1,
        opt_batch: int = 1,
        max_batch: int = 4,
    ) -> Path:
        """ONNXモデルからTensorRTエンジンをビルド・エクスポートする.

        Args:
            onnx_path: 入力ONNXモデルのパス.
            output_path: 出力エンジンファイル (.engine) のパス.
            input_size: 入力サイズ (height, width).
            min_batch: 最小バッチサイズ.
            opt_batch: 最適バッチサイズ.
            max_batch: 最大バッチサイズ.

        Returns:
            出力エンジンファイルのパス.

        Raises:
            FileNotFoundError: ONNXファイルが存在しない場合.
            RuntimeError: エンジンのビルドに失敗した場合.
        """
        onnx_file = Path(onnx_path)
        if not onnx_file.exists():
            raise FileNotFoundError(f"ONNXファイルが見つかりません: {onnx_file}")

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"TensorRTエンジンのビルドを開始します: {onnx_file} -> {out_file}")
        logger.info(
            f"入力サイズ: {input_size}, Batch (min={min_batch}, opt={opt_batch}, max={max_batch})"
        )

        builder = trt.Builder(self.trt_logger)

        # TRT 10.x 以降では EXPLICIT_BATCH がデフォルトとなりフラグが非推奨 / 削除されるため,
        # 属性の存在チェックを行って互換性を維持する.
        network_flags = 0
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
            network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)

        parser = trt.OnnxParser(network, self.trt_logger)
        config = builder.create_builder_config()

        # 推奨されるメモリプール制限 (ここでは4GB) の設定
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024
        )

        logger.debug("ONNXモデルのパース中...")
        with open(onnx_file, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("ONNXモデルのパースに失敗しました.")

        # 入力テンソルの取得
        if network.num_inputs == 0:
            raise RuntimeError("ネットワークに入力テンソルが見つかりません.")

        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        channels = input_tensor.shape[1]

        logger.debug(
            f"Optimization Profileの構築中... "
            f"(入力名: {input_name}, channels: {channels})"
        )
        profile = builder.create_optimization_profile()
        h, w = input_size

        # 形状設定: (batch, channels, height, width)
        profile.set_shape(
            input_name,
            (min_batch, channels, h, w),
            (opt_batch, channels, h, w),
            (max_batch, channels, h, w),
        )
        config.add_optimization_profile(profile)

        logger.info("エンジンをビルド中... (この処理には数分かかる場合があります)")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("TensorRTエンジンのビルドに失敗しました.")

        with open(out_file, "wb") as f:
            f.write(engine_bytes)

        logger.info(f"TensorRTエンジンビルド完了: {out_file}")

        file_size_mb = out_file.stat().st_size / (1024 * 1024)
        logger.debug(f"ファイルサイズ: {file_size_mb:.2f} MB")

        return out_file
