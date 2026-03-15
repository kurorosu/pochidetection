"""TensorRT INT8 キャリブレーション機能を提供するモジュール."""

import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

try:
    import tensorrt as trt

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False

from pochidetection.logging import LoggerManager

logger: logging.Logger = LoggerManager().get_logger(__name__)

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})


class INT8Calibrator(trt.IInt8EntropyCalibrator2 if _TRT_AVAILABLE else object):  # type: ignore[misc]
    """TensorRT INT8 Post-Training Quantization 用キャリブレータ.

    検証画像を順次読み込み, TensorRT ビルダーにキャリブレーションデータを供給する.
    ``IInt8EntropyCalibrator2`` を継承し, エントロピーベースのキャリブレーションを行う.

    Attributes:
        _image_paths: キャリブレーション用画像パスのリスト.
        _batch_size: キャリブレーションバッチサイズ.
        _current_index: 現在の読み込み位置.
        _device_buffer: GPU 上のキャリブレーションバッファ.
        _cache_path: キャリブレーションキャッシュの保存先.
        _transform: 画像前処理パイプライン.
    """

    def __init__(
        self,
        image_dir: Path | str,
        input_size: tuple[int, int],
        batch_size: int = 1,
        max_images: int | None = None,
        cache_path: Path | str | None = None,
    ) -> None:
        """初期化.

        Args:
            image_dir: キャリブレーション用画像ディレクトリ.
            input_size: 入力サイズ (height, width).
            batch_size: キャリブレーションバッチサイズ.
            max_images: 使用する最大画像数. None の場合は全画像を使用.
            cache_path: キャリブレーションキャッシュの保存先.
                None の場合はキャッシュを使用しない.

        Raises:
            ImportError: tensorrt がインストールされていない場合.
            FileNotFoundError: 画像ディレクトリが存在しない場合.
            ValueError: 画像ディレクトリに画像が見つからない場合.
        """
        if not _TRT_AVAILABLE:
            raise ImportError(
                "tensorrt パッケージがインストールされていません. "
                "GPU環境構築手順に従って TensorRT をインストールしてください."
            )

        super().__init__()

        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"キャリブレーション用画像ディレクトリが見つかりません: {image_dir}"
            )

        # 画像ファイルの収集
        image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"キャリブレーション用画像が見つかりません: {image_dir}")

        if max_images is not None:
            image_paths = image_paths[:max_images]

        self._image_paths = image_paths
        self._batch_size = batch_size
        self._current_index = 0
        self._cache_path = Path(cache_path) if cache_path is not None else None

        # SsdCocoDataset と同じ前処理パイプライン
        h, w = input_size
        self._transform = v2.Compose(
            [
                v2.Resize((h, w)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        # GPU バッファの事前確保 (batch_size, 3, H, W)
        self._device_buffer = torch.zeros(
            (batch_size, 3, h, w), dtype=torch.float32, device="cuda"
        )

        logger.info(
            f"INT8 キャリブレータを初期化しました: "
            f"{len(self._image_paths)} 枚, batch_size={batch_size}"
        )

    def get_batch_size(self) -> int:
        """キャリブレーションバッチサイズを返す.

        Returns:
            バッチサイズ.
        """
        return self._batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        """次のキャリブレーションバッチを供給する.

        Args:
            names: 入力テンソル名のリスト (TensorRT から渡される).

        Returns:
            GPU バッファのデバイスポインタリスト. 全画像を消費した場合は None.
        """
        if self._current_index >= len(self._image_paths):
            return None

        batch_end = min(self._current_index + self._batch_size, len(self._image_paths))
        actual_batch = batch_end - self._current_index

        for i in range(actual_batch):
            image_path = self._image_paths[self._current_index + i]
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            tensor = self._transform(image)
            self._device_buffer[i].copy_(tensor)

        self._current_index = batch_end

        if self._current_index % 50 == 0 or self._current_index >= len(
            self._image_paths
        ):
            logger.info(
                f"キャリブレーション進捗: "
                f"{self._current_index}/{len(self._image_paths)}"
            )

        return [self._device_buffer.data_ptr()]

    def read_calibration_cache(self) -> bytes | None:
        """キャリブレーションキャッシュを読み込む.

        Returns:
            キャッシュデータ. キャッシュが存在しない場合は None.
        """
        if self._cache_path is not None and self._cache_path.exists():
            logger.info(
                f"キャリブレーションキャッシュを読み込みます: {self._cache_path}"
            )
            return self._cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """キャリブレーションキャッシュを保存する.

        Args:
            cache: キャッシュデータ.
        """
        if self._cache_path is not None:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_bytes(cache)
            logger.info(
                f"キャリブレーションキャッシュを保存しました: {self._cache_path}"
            )
