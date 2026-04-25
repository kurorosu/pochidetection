"""orchestration/batch_inference.py のテスト (CLI batch フロー系)."""

from pathlib import Path
from typing import Any

from PIL import Image

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.core.detection import Detection
from pochidetection.interfaces.pipeline import IDetectionPipeline, ImageInput
from pochidetection.orchestration.batch_inference import (
    _collect_image_files,
    _run_inference,
)
from pochidetection.pipelines.context import PipelineContext
from pochidetection.reporting import InferenceSaver, Visualizer
from pochidetection.utils import PhasedTimer


class TestCollectImageFiles:
    """_collect_image_files のテスト."""

    def test_returns_none_when_dir_not_exists(self, tmp_path: Path) -> None:
        """ディレクトリが存在しない場合 None を返すことを確認."""
        result = _collect_image_files(str(tmp_path / "nonexistent"))

        assert result is None

    def test_returns_none_when_no_images(self, tmp_path: Path) -> None:
        """画像ファイルが無い場合 None を返すことを確認."""
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b")

        result = _collect_image_files(str(tmp_path))

        assert result is None

    def test_returns_image_files(self, tmp_path: Path) -> None:
        """画像ファイルを正しく収集することを確認."""
        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8")
        (tmp_path / "diagram.png").write_bytes(b"\x89PNG")
        (tmp_path / "notes.txt").write_text("not an image")

        result = _collect_image_files(str(tmp_path))

        assert result is not None
        assert len(result) == 2
        names = {f.name for f in result}
        assert names == {"photo.jpg", "diagram.png"}

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        """大文字拡張子の画像ファイルも収集することを確認."""
        (tmp_path / "photo.JPG").write_bytes(b"\xff\xd8")
        (tmp_path / "image.PNG").write_bytes(b"\x89PNG")

        result = _collect_image_files(str(tmp_path))

        assert result is not None
        assert len(result) == 2


class _StubPipeline(IDetectionPipeline[Any, Any]):
    """run() が空検出を返し, 付随する phased_timer もダミーで持つ stub."""

    def __init__(self, phased_timer: PhasedTimer) -> None:
        super().__init__()
        self._validate_phased_timer(phased_timer)

    def run(
        self, image: ImageInput, *, threshold: float | None = None
    ) -> list[Detection]:
        """空の検出リストを返す."""
        return []


def _build_stub_context(saver_base: Path) -> PipelineContext:
    """_run_inference テスト用の最小 PipelineContext を構築する."""
    timer = PhasedTimer(
        phases=["preprocess", "inference", "postprocess"],
        device="cpu",
        skip_first=False,
    )
    pipeline = _StubPipeline(timer)
    saver = InferenceSaver(saver_base)
    visualizer = Visualizer()
    return PipelineContext(
        pipeline=pipeline,
        phased_timer=timer,
        visualizer=visualizer,
        saver=saver,
        label_mapper=None,
        class_names=None,
        actual_device="cpu",
        precision="fp32",
    )


class TestRunInferenceInferDebug:
    """_run_inference の infer_debug 保存動作 (E2E 寄り)."""

    def _prepare_images(self, dir_path: Path, n: int) -> list[Path]:
        """jpg を n 枚作成してパスリストを返す."""
        dir_path.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for i in range(n):
            p = dir_path / f"img_{i:03d}.jpg"
            Image.new("RGB", (128, 32), color=(i * 60, 0, 0)).save(p, format="JPEG")
            files.append(p)
        return files

    def test_saves_first_n_images_to_inference_output_dir(self, tmp_path: Path) -> None:
        """save_count=2 なら先頭 2 枚のみ ``<output_dir>/infer_debug/`` に保存."""
        saver_base = tmp_path / "model"
        saver_base.mkdir()
        image_files = self._prepare_images(tmp_path / "images", 3)

        ctx = _build_stub_context(saver_base)
        config: DetectionConfigDict = {
            "infer_debug_save_count": 2,
            "image_size": {"height": 64, "width": 64},
            "letterbox": True,
        }

        _run_inference(image_files, ctx, config, save_crop=False)

        debug_dir = ctx.saver.output_dir / "infer_debug"
        saved = sorted(p.name for p in debug_dir.iterdir() if p.suffix == ".jpg")
        assert saved == ["infer_0000.jpg", "infer_0001.jpg"]

    def test_no_save_when_disabled(self, tmp_path: Path) -> None:
        """infer_debug_save_count=0 で ``infer_debug/`` が作られない."""
        saver_base = tmp_path / "model"
        saver_base.mkdir()
        image_files = self._prepare_images(tmp_path / "images", 2)

        ctx = _build_stub_context(saver_base)
        config: DetectionConfigDict = {
            "infer_debug_save_count": 0,
            "image_size": {"height": 64, "width": 64},
            "letterbox": True,
        }

        _run_inference(image_files, ctx, config, save_crop=False)

        assert not (ctx.saver.output_dir / "infer_debug").exists()
