"""COCO プリトレインモデル関連のテスト."""

from pathlib import Path

from pochidetection.core.coco_classes import (
    COCO_CLASS_NAMES,
    COCO_NUM_CLASSES,
    PRETRAINED_CONFIG_PATH,
)
from pochidetection.pipelines.model_path import PRETRAINED, resolve_model_path
from pochidetection.utils import ConfigLoader


class TestCocoClassNames:
    """COCO クラス定数のテスト."""

    def test_num_classes_is_80(self) -> None:
        """COCO_NUM_CLASSES が 80 である."""
        assert COCO_NUM_CLASSES == 80

    def test_class_names_length_matches(self) -> None:
        """COCO_CLASS_NAMES の要素数が COCO_NUM_CLASSES と一致する."""
        assert len(COCO_CLASS_NAMES) == COCO_NUM_CLASSES

    def test_first_class_is_person(self) -> None:
        """先頭クラスが person である."""
        assert COCO_CLASS_NAMES[0] == "person"

    def test_last_class_is_toothbrush(self) -> None:
        """末尾クラスが toothbrush である."""
        assert COCO_CLASS_NAMES[-1] == "toothbrush"


class TestPretrainedConfig:
    """プリトレイン用 config ファイルのテスト."""

    def test_architecture_is_rtdetr(self) -> None:
        """アーキテクチャが RTDetr 固定である."""
        config = ConfigLoader.load(PRETRAINED_CONFIG_PATH)
        assert config["architecture"] == "RTDetr"

    def test_num_classes_is_80(self) -> None:
        """num_classes が 80 である."""
        config = ConfigLoader.load(PRETRAINED_CONFIG_PATH)
        assert config["num_classes"] == 80

    def test_class_names_matches_coco(self) -> None:
        """class_names が COCO_CLASS_NAMES と一致する."""
        config = ConfigLoader.load(PRETRAINED_CONFIG_PATH)
        assert config["class_names"] == COCO_CLASS_NAMES

    def test_has_local_files_only(self) -> None:
        """local_files_only フィールドが存在する."""
        config = ConfigLoader.load(PRETRAINED_CONFIG_PATH)
        assert "local_files_only" in config


class TestResolveModelPathPretrained:
    """resolve_model_path のプリトレインフォールバックテスト."""

    def test_returns_pretrained_when_no_workspace(self, tmp_path: Path) -> None:
        """ワークスペースが存在しない場合 PRETRAINED を返す."""
        config = {"work_dir": str(tmp_path / "empty_work_dirs")}
        result = resolve_model_path(config, None)  # type: ignore[arg-type]
        assert result == PRETRAINED

    def test_returns_explicit_model_path(self, tmp_path: Path) -> None:
        """明示的にモデルパスを指定した場合はそのパスを返す."""
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        result = resolve_model_path({}, str(model_dir))  # type: ignore[arg-type]
        assert result == model_dir

    def test_returns_none_when_explicit_path_not_exists(self) -> None:
        """存在しないパスを指定した場合は None を返す."""
        result = resolve_model_path({}, "/nonexistent/path")  # type: ignore[arg-type]
        assert result is None
