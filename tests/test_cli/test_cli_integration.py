"""CLI 統合テスト.

引数パース, アーキテクチャ別ディスパッチ, コマンド間のパス引き継ぎを検証する.
"""

from pathlib import Path

import pytest

from pochidetection.cli.commands.infer import (
    is_rtsp_source,
    is_video_file,
    is_webcam_source,
    run_infer,
)
from pochidetection.cli.parser import _create_parser
from pochidetection.cli.registry import get_infer_for_arch, get_train_for_arch

# ------------------------------------------------------------------ #
# 引数パース
# ------------------------------------------------------------------ #


class TestTrainArgParse:
    """train サブコマンドの引数パーステスト."""

    def test_default_config(self) -> None:
        """config 未指定時にデフォルト値が設定されることを確認する."""
        parser = _create_parser()
        args = parser.parse_args(["train"])
        assert args.command == "train"
        assert args.config == "configs/rtdetr_coco.py"

    def test_custom_config(self) -> None:
        """config 指定時にその値が使われることを確認する."""
        parser = _create_parser()
        args = parser.parse_args(["train", "-c", "configs/ssdlite_coco.py"])
        assert args.config == "configs/ssdlite_coco.py"

    def test_debug_flag(self) -> None:
        """--debug フラグが正しくパースされることを確認する."""
        parser = _create_parser()
        args = parser.parse_args(["--debug", "train"])
        assert args.debug is True

    def test_debug_flag_default(self) -> None:
        """--debug 未指定時に False であることを確認する."""
        parser = _create_parser()
        args = parser.parse_args(["train"])
        assert args.debug is False


class TestInferArgParse:
    """infer サブコマンドの引数パーステスト."""

    def test_defaults(self) -> None:
        """全オプション未指定時のデフォルト値を確認する."""
        parser = _create_parser()
        args = parser.parse_args(["infer"])
        assert args.command == "infer"
        assert args.dir is None
        assert args.model_dir is None
        assert args.config is None

    def test_all_options(self) -> None:
        """全オプション指定時の値を確認する."""
        parser = _create_parser()
        args = parser.parse_args(
            [
                "infer",
                "-d",
                "/images",
                "-m",
                "/model/best",
                "-c",
                "my_config.py",
            ]
        )
        assert args.dir == "/images"
        assert args.model_dir == "/model/best"
        assert args.config == "my_config.py"


class TestExportArgParse:
    """export サブコマンドの引数パーステスト."""

    def test_required_model_path(self) -> None:
        """-m 未指定で SystemExit が発生することを確認する."""
        parser = _create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["export"])

    def test_defaults(self) -> None:
        """オプション未指定時のデフォルト値を確認する."""
        parser = _create_parser()
        args = parser.parse_args(["export", "-m", "/model/best"])
        assert args.command == "export"
        assert args.model_path == "/model/best"
        assert args.output is None
        assert args.input_size is None
        assert args.opset_version == 17
        assert args.skip_verify is False
        assert args.fp16 is False
        assert args.min_batch == 1
        assert args.opt_batch == 1
        assert args.max_batch == 4

    def test_input_size_parsing(self) -> None:
        """--input-size が 2 つの整数としてパースされることを確認する."""
        parser = _create_parser()
        args = parser.parse_args(
            [
                "export",
                "-m",
                "/model",
                "--input-size",
                "320",
                "320",
            ]
        )
        assert args.input_size == [320, 320]

    def test_all_flags(self) -> None:
        """全フラグ指定時の値を確認する."""
        parser = _create_parser()
        args = parser.parse_args(
            [
                "export",
                "-m",
                "/model",
                "--fp16",
                "--skip-verify",
                "--int8",
                "--opset-version",
                "14",
                "--min-batch",
                "2",
                "--opt-batch",
                "4",
                "--max-batch",
                "8",
            ]
        )
        assert args.fp16 is True
        assert args.skip_verify is True
        assert args.int8 is True
        assert args.opset_version == 14
        assert args.min_batch == 2
        assert args.opt_batch == 4
        assert args.max_batch == 8


class TestNoCommand:
    """サブコマンド未指定時のパーステスト."""

    def test_no_command_raises_system_exit(self) -> None:
        """サブコマンドなしでパースすると SystemExit が発生することを確認する."""
        parser = _create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([])
        assert exc_info.value.code == 2


# ------------------------------------------------------------------ #
# アーキテクチャ別ディスパッチ
# ------------------------------------------------------------------ #


class TestGetTrainForArchDispatch:
    """get_train_for_arch のアーキテクチャ別ディスパッチテスト."""

    def test_rtdetr_default(self) -> None:
        """architecture 未指定時に RT-DETR の train が返ることを確認する."""
        from pochidetection.scripts.rtdetr.train import train as rtdetr_train

        fn = get_train_for_arch({})
        assert fn is rtdetr_train

    def test_rtdetr_explicit(self) -> None:
        """architecture='RTDetr' で RT-DETR の train が返ることを確認する."""
        from pochidetection.scripts.rtdetr.train import train as rtdetr_train

        fn = get_train_for_arch({"architecture": "RTDetr"})
        assert fn is rtdetr_train

    def test_ssdlite(self) -> None:
        """architecture='SSDLite' で SSDLite の train が返ることを確認する."""
        from pochidetection.scripts.ssdlite.train import train as ssdlite_train

        fn = get_train_for_arch({"architecture": "SSDLite"})
        assert fn is ssdlite_train

    def test_ssd300(self) -> None:
        """architecture='SSD300' で SSD300 の train が返ることを確認する."""
        from pochidetection.scripts.ssd300.train import train as ssd300_train

        fn = get_train_for_arch({"architecture": "SSD300"})
        assert fn is ssd300_train


class TestGetInferForArchDispatch:
    """get_infer_for_arch のアーキテクチャ別ディスパッチテスト."""

    def test_rtdetr_default(self) -> None:
        """architecture 未指定時に RT-DETR の infer が返ることを確認する."""
        from pochidetection.scripts.rtdetr.infer import infer as rtdetr_infer

        fn = get_infer_for_arch({})
        assert fn is rtdetr_infer

    def test_rtdetr_explicit(self) -> None:
        """architecture='RTDetr' で RT-DETR の infer が返ることを確認する."""
        from pochidetection.scripts.rtdetr.infer import infer as rtdetr_infer

        fn = get_infer_for_arch({"architecture": "RTDetr"})
        assert fn is rtdetr_infer

    def test_ssdlite(self) -> None:
        """architecture='SSDLite' で SSDLite の infer が返ることを確認する."""
        from pochidetection.scripts.ssdlite.infer import infer as ssdlite_infer

        fn = get_infer_for_arch({"architecture": "SSDLite"})
        assert fn is ssdlite_infer

    def test_ssd300(self) -> None:
        """architecture='SSD300' で SSD300 の infer が返ることを確認する."""
        from pochidetection.scripts.ssd300.infer import infer as ssd300_infer

        fn = get_infer_for_arch({"architecture": "SSD300"})
        assert fn is ssd300_infer


# ------------------------------------------------------------------ #
# コマンド間パス引き継ぎ
# ------------------------------------------------------------------ #


class TestConfigResolver:
    """train 出力ディレクトリが infer / export の入力として利用可能であることのテスト."""

    def test_config_in_model_dir_is_resolved(self, tmp_path: Path) -> None:
        """モデルディレクトリ内の config.py が自動解決されることを確認する."""
        from pochidetection.utils.config_resolver import resolve_config_path

        # train がワークスペースに config.py をコピーする構造を模倣
        workspace = tmp_path / "work_dirs" / "20260314_001"
        best_dir = workspace / "best"
        best_dir.mkdir(parents=True)

        config_file = workspace / "config.py"
        config_file.write_text(
            'data_root = "data"\nnum_classes = 4\n', encoding="utf-8"
        )

        resolved = resolve_config_path(
            config=None,
            model_dir=str(best_dir),
            default_config="configs/rtdetr_coco.py",
        )
        assert resolved == str(config_file)

    def test_explicit_config_takes_priority(self, tmp_path: Path) -> None:
        """明示的な config 指定が自動解決より優先されることを確認する."""
        from pochidetection.utils.config_resolver import resolve_config_path

        resolved = resolve_config_path(
            config="my_config.py",
            model_dir=str(tmp_path),
            default_config="configs/rtdetr_coco.py",
        )
        assert resolved == "my_config.py"

    def test_fallback_to_default_config(self, tmp_path: Path) -> None:
        """config 未指定かつ model_dir に .py がない場合にデフォルトにフォールバックすることを確認する."""
        from pochidetection.utils.config_resolver import resolve_config_path

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        resolved = resolve_config_path(
            config=None,
            model_dir=str(empty_dir),
            default_config="configs/rtdetr_coco.py",
        )
        assert resolved == "configs/rtdetr_coco.py"

    def test_onnx_file_resolves_config_from_parent(self, tmp_path: Path) -> None:
        """ONNX ファイル指定時に同ディレクトリの config.py が解決されることを確認する."""
        from pochidetection.utils.config_resolver import resolve_config_path

        model_dir = tmp_path / "best"
        model_dir.mkdir()
        onnx_file = model_dir / "model.onnx"
        onnx_file.touch()
        config_file = model_dir / "config.py"
        config_file.write_text(
            'data_root = "data"\nnum_classes = 4\n', encoding="utf-8"
        )

        resolved = resolve_config_path(
            config=None,
            model_dir=str(onnx_file),
            default_config="configs/rtdetr_coco.py",
        )
        assert resolved == str(config_file)


# ------------------------------------------------------------------ #
# run_infer エラーハンドリング
# ------------------------------------------------------------------ #


class TestRunInferValidation:
    """run_infer のバリデーションテスト."""

    def test_no_image_dir_exits(self, tmp_path: Path) -> None:
        """画像ディレクトリ未指定時に SystemExit が発生することを確認する."""
        config_file = tmp_path / "config.py"
        config_file.write_text(
            'architecture = "RTDetr"\ndata_root = "data"\nnum_classes = 4\n',
            encoding="utf-8",
        )

        parser = _create_parser()
        args = parser.parse_args(["infer", "-c", str(config_file)])

        with pytest.raises(SystemExit):
            run_infer(args)


# ------------------------------------------------------------------ #
# 動画ファイル判別
# ------------------------------------------------------------------ #


class TestIsVideoFile:
    """is_video_file のテスト."""

    def test_mp4(self) -> None:
        """mp4 が動画と判定される."""
        assert is_video_file("video.mp4") is True

    def test_avi(self) -> None:
        """avi が動画と判定される."""
        assert is_video_file("video.avi") is True

    def test_mov(self) -> None:
        """mov が動画と判定される."""
        assert is_video_file("video.mov") is True

    def test_uppercase(self) -> None:
        """大文字拡張子も動画と判定される."""
        assert is_video_file("video.MP4") is True

    def test_image_dir(self) -> None:
        """ディレクトリパスは動画でない."""
        assert is_video_file("images/") is False

    def test_image_file(self) -> None:
        """画像ファイルは動画でない."""
        assert is_video_file("image.jpg") is False


class TestIntervalArgParse:
    """--interval 引数のパーステスト."""

    def test_default_interval(self) -> None:
        """interval 未指定時にデフォルト値 1 が設定される."""
        parser = _create_parser()
        args = parser.parse_args(["infer", "-d", "images/"])
        assert args.interval == 1

    def test_custom_interval(self) -> None:
        """interval 指定時にその値が使われる."""
        parser = _create_parser()
        args = parser.parse_args(["infer", "-d", "video.mp4", "--interval", "3"])
        assert args.interval == 3


# ------------------------------------------------------------------ #
# Webcam / RTSP 判別
# ------------------------------------------------------------------ #


class TestIsWebcamSource:
    """is_webcam_source のテスト."""

    def test_device_id_zero(self) -> None:
        """'0' は Webcam デバイス ID."""
        assert is_webcam_source("0") is True

    def test_device_id_number(self) -> None:
        """'2' は Webcam デバイス ID."""
        assert is_webcam_source("2") is True

    def test_rtsp_url(self) -> None:
        """RTSP URL は Webcam でない."""
        assert is_webcam_source("rtsp://192.168.1.10/stream") is False

    def test_video_file(self) -> None:
        """動画ファイルは Webcam でない."""
        assert is_webcam_source("video.mp4") is False

    def test_image_dir(self) -> None:
        """ディレクトリパスは Webcam でない."""
        assert is_webcam_source("images/") is False


class TestIsRtspSource:
    """is_rtsp_source のテスト."""

    def test_rtsp_url(self) -> None:
        """rtsp:// で始まる URL は RTSP."""
        assert is_rtsp_source("rtsp://192.168.1.10/stream") is True

    def test_http_url(self) -> None:
        """http:// で始まる URL は RTSP (HTTP ストリーム)."""
        assert is_rtsp_source("http://192.168.1.10/stream") is True

    def test_device_id(self) -> None:
        """デバイス ID は RTSP でない."""
        assert is_rtsp_source("0") is False

    def test_video_file(self) -> None:
        """動画ファイルは RTSP でない."""
        assert is_rtsp_source("video.mp4") is False

    def test_image_dir(self) -> None:
        """ディレクトリパスは RTSP でない."""
        assert is_rtsp_source("images/") is False


# ------------------------------------------------------------------ #
# --record 引数パース
# ------------------------------------------------------------------ #


class TestRecordArgParse:
    """--record 引数のパーステスト."""

    def test_default_record(self) -> None:
        """record 未指定時に False."""
        parser = _create_parser()
        args = parser.parse_args(["infer", "-d", "0"])
        assert args.record is False

    def test_record_flag(self) -> None:
        """record 指定時に True."""
        parser = _create_parser()
        args = parser.parse_args(["infer", "-d", "0", "--record"])
        assert args.record is True
