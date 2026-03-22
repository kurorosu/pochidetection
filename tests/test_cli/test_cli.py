"""物体検出 CLI のテスト."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


class TestCliNoCommand:
    """サブコマンド未指定時の動作テスト."""

    def test_no_command_exits_with_error(self) -> None:
        """サブコマンドなしで実行するとエラーメッセージが表示され非ゼロ終了することを確認."""
        result = subprocess.run(
            [sys.executable, "-m", "pochidetection.cli.main"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "command" in result.stderr.lower()


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


class TestCliExportGuard:
    """export コマンドのガードテスト."""

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

    def test_export_onnx_dispatches_to_trt(
        self, ssdlite_config: Path, tmp_path: Path
    ) -> None:
        """export -m model.onnx で TRT ルートに入ることを確認.

        ONNX ファイルが空のためエクスポート自体は失敗するが,
        CLI ルーティングが .onnx を TRT として受け付けることを検証する.
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.touch()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pochidetection.cli.main",
                "export",
                "-m",
                str(onnx_file),
                "-c",
                str(ssdlite_config),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "対応していません" not in result.stderr

    def test_export_fp16_with_rtdetr_config_exits_with_error(
        self, tmp_path: Path
    ) -> None:
        """RT-DETR config でフォルダ指定 + --fp16 指定時にエラー終了することを確認."""
        config = tmp_path / "rtdetr_config.py"
        config.write_text(
            'data_root = "data"\n' "num_classes = 4\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pochidetection.cli.main",
                "export",
                "-m",
                str(tmp_path),
                "-c",
                str(config),
                "--fp16",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 1
        assert "--fp16" in result.stderr
