"""prepare-demo の dispatcher 単体テスト.

実 HF DL や TRT engine ビルドは含めず, build_demo_config と TRT 不在時の
副作用ゼロ exit のみを検証する.
"""

import argparse
import sys
from pathlib import Path

import pytest

from pochidetection.cli.commands.prepare_demo import (
    DEMO_MODEL_NAME,
    build_demo_config,
    run_prepare_demo,
)
from pochidetection.core.coco_classes import COCO_CLASS_NAMES, COCO_NUM_CLASSES


class TestBuildDemoConfig:
    """build_demo_config が schema 必須キーを満たし image_size が反映されること."""

    def test_returns_minimum_required_keys(self) -> None:
        """num_classes / data_root を含み model_name は r50vd."""
        config = build_demo_config({"height": 640, "width": 640})
        assert config["architecture"] == "RTDetr"
        assert config["model_name"] == DEMO_MODEL_NAME
        assert config["num_classes"] == COCO_NUM_CLASSES
        assert config["class_names"] == list(COCO_CLASS_NAMES)
        assert config["data_root"] == "."

    def test_image_size_propagates(self) -> None:
        """--input-size 値がそのまま image_size に乗る."""
        config = build_demo_config({"height": 800, "width": 1280})
        assert config["image_size"] == {"height": 800, "width": 1280}

    def test_validates_against_pydantic_schema(self) -> None:
        """ConfigLoader が validate する Pydantic schema (extra=forbid) を通る."""
        from pochidetection.configs.schemas import DetectionConfig

        config = build_demo_config({"height": 640, "width": 640})
        DetectionConfig.model_validate(config)


class TestTensorrtAbsentExit:
    """TensorRT 未インストール時に副作用ゼロで exit すること."""

    def test_exits_without_creating_workspace(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TRT 不在で sys.exit(1) かつ work_dirs/ 配下に何も作られない."""
        monkeypatch.chdir(tmp_path)
        # CPython の sys.modules[None] sentinel で ImportError を強制
        monkeypatch.setitem(sys.modules, "pochidetection.tensorrt", None)

        args = argparse.Namespace(input_size=[640, 640])
        with pytest.raises(SystemExit) as exc_info:
            run_prepare_demo(args)

        assert exc_info.value.code == 1
        assert not (tmp_path / "work_dirs").exists()
