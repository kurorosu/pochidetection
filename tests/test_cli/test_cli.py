"""物体検出 CLI のテスト."""

import subprocess
import sys
from pathlib import Path

import pytest


class TestCliNoCommand:
    """サブコマンド未指定時の動作テスト."""

    def test_no_command_shows_help_without_error(self) -> None:
        """サブコマンドなしで実行するとヘルプが表示され正常終了することを確認."""
        result = subprocess.run(
            [sys.executable, "-m", "pochidetection.cli.main"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "使用例" in result.stdout


class TestCliInferImageDir:
    """infer コマンドの -d / infer_image_dir フォールバックテスト."""

    @pytest.fixture
    def config_with_infer_image_dir(self, tmp_path: Path) -> Path:
        """infer_image_dir 付き設定ファイルを作成する fixture."""
        config = tmp_path / "config.py"
        config.write_text(
            'data_root = "data"\n'
            "num_classes = 4\n"
            f'infer_image_dir = "{tmp_path / "images"}"\n',
            encoding="utf-8",
        )
        return config

    @pytest.fixture
    def config_without_infer_image_dir(self, tmp_path: Path) -> Path:
        """infer_image_dir なし設定ファイルを作成する fixture."""
        config = tmp_path / "config_no_dir.py"
        config.write_text(
            'data_root = "data"\n' "num_classes = 4\n",
            encoding="utf-8",
        )
        return config

    def test_no_d_and_no_config_infer_image_dir_exits_with_error(
        self, config_without_infer_image_dir: Path
    ) -> None:
        """-d 未指定かつ config に infer_image_dir がない場合にエラー終了."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pochidetection.cli.main",
                "infer",
                "-c",
                str(config_without_infer_image_dir),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 1
        assert "infer_image_dir" in result.stderr

    def test_d_flag_is_optional(self) -> None:
        """-d なしでも argparse エラーにならないことを確認."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pochidetection.cli.main",
                "infer",
                "-c",
                "configs/rtdetr_coco.py",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # config に infer_image_dir があるので argparse エラーではない.
        # 推論自体は失敗しうるが, argparse の "required" エラーは出ない.
        assert "error: the following arguments are required: -d" not in result.stderr


class TestCliExportSsdliteGuard:
    """export / export-trt コマンドの SSDLite ガードテスト."""

    @pytest.fixture
    def ssdlite_config(self, tmp_path: Path) -> Path:
        """SSDLite 設定ファイルを作成する fixture."""
        config = tmp_path / "ssdlite_config.py"
        config.write_text(
            'architecture = "SSDLite"\n'
            "num_classes = 4\n"
            'image_size = {"height": 320, "width": 320}\n'
            'data_root = "data"\n',
            encoding="utf-8",
        )
        return config

    def test_export_with_ssdlite_config_exits_with_error(
        self, ssdlite_config: Path, tmp_path: Path
    ) -> None:
        """SSDLite config で export 実行時にエラー終了することを確認."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pochidetection.cli.main",
                "export",
                "-m",
                str(tmp_path),
                "-c",
                str(ssdlite_config),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 1
        assert "SSDLite" in result.stderr
        assert "対応していません" in result.stderr

    def test_export_trt_with_ssdlite_config_exits_with_error(
        self, ssdlite_config: Path, tmp_path: Path
    ) -> None:
        """SSDLite config で export-trt 実行時にエラー終了することを確認."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.touch()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pochidetection.cli.main",
                "export-trt",
                "-i",
                str(onnx_file),
                "-c",
                str(ssdlite_config),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 1
        assert "SSDLite" in result.stderr
        assert "対応していません" in result.stderr
