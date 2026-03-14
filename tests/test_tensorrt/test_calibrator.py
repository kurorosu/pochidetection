"""INT8Calibrator のテスト."""

from pathlib import Path

import pytest

pytest.importorskip("tensorrt")

from pochidetection.tensorrt import INT8Calibrator

from .conftest import CALIB_NUM_IMAGES, INPUT_SIZE


class TestINT8CalibratorInit:
    """INT8Calibrator の初期化テスト."""

    def test_init_with_valid_dir(self, calib_image_dir: Path) -> None:
        """有効な画像ディレクトリで正常に初期化できることを確認."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
        )
        assert calibrator is not None

    def test_batch_size_default(self, calib_image_dir: Path) -> None:
        """デフォルトのバッチサイズが 1 であることを確認."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
        )
        assert calibrator.get_batch_size() == 1

    def test_batch_size_custom(self, calib_image_dir: Path) -> None:
        """カスタムバッチサイズが正しく設定されることを確認."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
            batch_size=2,
        )
        assert calibrator.get_batch_size() == 2

    def test_max_images_limits_count(self, calib_image_dir: Path) -> None:
        """max_images で画像数が制限されることを確認."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
            max_images=2,
        )
        assert len(calibrator._image_paths) == 2

    def test_all_images_loaded_without_limit(self, calib_image_dir: Path) -> None:
        """max_images 未指定で全画像がロードされることを確認."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
        )
        assert len(calibrator._image_paths) == CALIB_NUM_IMAGES

    def test_init_nonexistent_dir_raises_error(self, tmp_path: Path) -> None:
        """存在しないディレクトリで FileNotFoundError が発生することを確認."""
        with pytest.raises(
            FileNotFoundError, match="キャリブレーション用画像ディレクトリ"
        ):
            INT8Calibrator(
                image_dir=tmp_path / "nonexistent",
                input_size=INPUT_SIZE,
            )

    def test_init_empty_dir_raises_error(self, tmp_path: Path) -> None:
        """画像のない空ディレクトリで ValueError が発生することを確認."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(
            ValueError, match="キャリブレーション用画像が見つかりません"
        ):
            INT8Calibrator(
                image_dir=empty_dir,
                input_size=INPUT_SIZE,
            )

    def test_init_accepts_str_path(self, calib_image_dir: Path) -> None:
        """文字列パスを受け入れることを確認."""
        calibrator = INT8Calibrator(
            image_dir=str(calib_image_dir),
            input_size=INPUT_SIZE,
        )
        assert calibrator is not None


class TestINT8CalibratorGetBatch:
    """INT8Calibrator.get_batch() のテスト."""

    def test_get_batch_returns_device_pointer(self, calib_image_dir: Path) -> None:
        """get_batch がデバイスポインタのリストを返すことを確認."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
        )
        result = calibrator.get_batch(["pixel_values"])
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], int)

    def test_get_batch_returns_none_after_exhaustion(
        self, calib_image_dir: Path
    ) -> None:
        """全画像を消費後に None を返すことを確認."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
            max_images=2,
        )
        # 2 枚消費
        result1 = calibrator.get_batch(["pixel_values"])
        assert result1 is not None
        result2 = calibrator.get_batch(["pixel_values"])
        assert result2 is not None
        # 全消費後
        result3 = calibrator.get_batch(["pixel_values"])
        assert result3 is None

    def test_get_batch_processes_all_images(self, calib_image_dir: Path) -> None:
        """全画像が正しく処理されることを確認."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
        )
        batch_count = 0
        while calibrator.get_batch(["pixel_values"]) is not None:
            batch_count += 1
        assert batch_count == CALIB_NUM_IMAGES


class TestINT8CalibratorCalibrationFlow:
    """キャリブレーションフローの統合テスト."""

    def test_multi_batch_sequence(self, calib_image_dir: Path) -> None:
        """複数バッチにわたる get_batch() シーケンスが正しく動作することを確認."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
            batch_size=2,
        )
        # 5 枚, batch_size=2 → ceil(5/2)=3 バッチ
        batch_count = 0
        while calibrator.get_batch(["pixel_values"]) is not None:
            batch_count += 1
        assert batch_count == 3

    def test_images_fewer_than_batch_size(self, calib_image_dir: Path) -> None:
        """画像枚数がバッチサイズ未満でも正しく動作することを確認."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
            batch_size=3,
            max_images=1,
        )
        result = calibrator.get_batch(["pixel_values"])
        assert result is not None
        # 1枚で完了
        assert calibrator.get_batch(["pixel_values"]) is None

    def test_cache_reuse_across_instances(
        self, calib_image_dir: Path, tmp_path: Path
    ) -> None:
        """キャッシュが別インスタンスで再利用できることを確認."""
        cache_path = tmp_path / "reuse_cache.bin"
        calibrator1 = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
            cache_path=cache_path,
        )
        test_data = b"calibration_cache_for_reuse"
        calibrator1.write_calibration_cache(test_data)

        # 別インスタンスで読み込み
        calibrator2 = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
            cache_path=cache_path,
        )
        assert calibrator2.read_calibration_cache() == test_data


class TestINT8CalibratorCache:
    """INT8Calibrator のキャッシュ機能テスト."""

    def test_read_cache_returns_none_when_no_cache(
        self, calib_image_dir: Path, tmp_path: Path
    ) -> None:
        """キャッシュが存在しない場合に None を返すことを確認."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
            cache_path=tmp_path / "nonexistent_cache.bin",
        )
        assert calibrator.read_calibration_cache() is None

    def test_write_and_read_cache(self, calib_image_dir: Path, tmp_path: Path) -> None:
        """キャッシュの書き込みと読み込みが正しく動作することを確認."""
        cache_path = tmp_path / "test_cache.bin"
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
            cache_path=cache_path,
        )

        test_data = b"dummy_calibration_cache_data"
        calibrator.write_calibration_cache(test_data)

        assert cache_path.exists()
        assert calibrator.read_calibration_cache() == test_data

    def test_no_cache_when_path_is_none(self, calib_image_dir: Path) -> None:
        """cache_path が None の場合にキャッシュを使用しないことを確認."""
        calibrator = INT8Calibrator(
            image_dir=calib_image_dir,
            input_size=INPUT_SIZE,
            cache_path=None,
        )
        assert calibrator.read_calibration_cache() is None
        # write しても例外が発生しない
        calibrator.write_calibration_cache(b"data")
