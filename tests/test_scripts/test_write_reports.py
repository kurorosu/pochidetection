"""write_reports() の統合テスト."""

import csv
import json
from pathlib import Path

from pochidetection.configs.schemas import DetectionConfigDict
from pochidetection.core.detection import Detection
from pochidetection.scripts.common.inference import PipelineContext, write_reports
from pochidetection.scripts.common.saver import InferenceSaver
from pochidetection.scripts.common.visualizer import Visualizer
from pochidetection.utils.phased_timer import PhasedTimer
from pochidetection.visualization import LabelMapper


def _make_context(tmp_path: Path, class_names: list[str]) -> PipelineContext:
    """テスト用 PipelineContext を構築する."""
    label_mapper = LabelMapper(class_names)
    visualizer = Visualizer(label_mapper=label_mapper)
    saver = InferenceSaver(tmp_path)
    phased_timer = PhasedTimer(
        phases=["preprocess", "inference", "postprocess"],
        device="cpu",
        skip_first=False,
    )
    # 各フェーズに計測データを1回分記録
    for phase in ["preprocess", "inference", "postprocess"]:
        with phased_timer.measure(phase):
            pass

    return PipelineContext(
        pipeline=None,  # type: ignore[arg-type]
        phased_timer=phased_timer,
        visualizer=visualizer,
        saver=saver,
        label_mapper=label_mapper,
        class_names=class_names,
        actual_device="cpu",
        precision="fp32",
    )


class TestWriteReportsNormal:
    """正常な検出結果での write_reports テスト."""

    def test_generates_all_report_files(
        self,
        tmp_path: Path,
        coco_annotation: Path,
        sample_predictions: dict[str, list[Detection]],
    ) -> None:
        """全レポートファイルが生成されることを確認."""
        config: DetectionConfigDict = {
            "annotation_path": str(coco_annotation),
            "class_names": ["cat", "dog"],
        }
        ctx = _make_context(tmp_path, ["cat", "dog"])
        image_files = [Path(f"images/{name}") for name in sample_predictions]
        model_path = tmp_path / "model"
        model_path.mkdir()

        write_reports(config, image_files, sample_predictions, ctx, model_path)

        output_dir = ctx.saver.output_dir
        assert (output_dir / "detection_summary.json").exists()
        assert (output_dir / "detection_results.csv").exists()
        assert (output_dir / "benchmark_result.json").exists()
        assert (output_dir / "confusion_matrix.html").exists()

    def test_csv_content(
        self,
        tmp_path: Path,
        coco_annotation: Path,
        sample_predictions: dict[str, list[Detection]],
    ) -> None:
        """CSV のカラム名と行数を確認."""
        config: DetectionConfigDict = {
            "annotation_path": str(coco_annotation),
            "class_names": ["cat", "dog"],
        }
        ctx = _make_context(tmp_path, ["cat", "dog"])
        image_files = [Path(f"images/{name}") for name in sample_predictions]
        model_path = tmp_path / "model"
        model_path.mkdir()

        write_reports(config, image_files, sample_predictions, ctx, model_path)

        csv_path = ctx.saver.output_dir / "detection_results.csv"
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert "image_name" in reader.fieldnames  # type: ignore[operator]
        assert "confidence" in reader.fieldnames  # type: ignore[operator]
        assert "status" in reader.fieldnames  # type: ignore[operator]
        # img001=2 TP, img002=1 TP, img003=1 FN (GT あり検出なし) → 4 rows
        assert len(rows) == 4

    def test_detection_summary_content(
        self,
        tmp_path: Path,
        sample_predictions: dict[str, list[Detection]],
    ) -> None:
        """detection_summary.json の内容を確認."""
        config: DetectionConfigDict = {}
        ctx = _make_context(tmp_path, ["cat", "dog"])
        image_files = [Path(f"images/{name}") for name in sample_predictions]
        model_path = tmp_path / "model"
        model_path.mkdir()

        write_reports(config, image_files, sample_predictions, ctx, model_path)

        summary_path = ctx.saver.output_dir / "detection_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["total_images"] == 3
        assert summary["total_detections"] == 3


class TestWriteReportsEmpty:
    """検出結果が空の場合の write_reports テスト."""

    def test_empty_predictions(self, tmp_path: Path) -> None:
        """全画像で検出0件のレポートが生成されることを確認."""
        config: DetectionConfigDict = {}
        ctx = _make_context(tmp_path, ["cat", "dog"])
        empty_predictions: dict[str, list[Detection]] = {
            "img001.jpg": [],
            "img002.jpg": [],
        }
        image_files = [Path("images/img001.jpg"), Path("images/img002.jpg")]
        model_path = tmp_path / "model"
        model_path.mkdir()

        write_reports(config, image_files, empty_predictions, ctx, model_path)

        output_dir = ctx.saver.output_dir
        assert (output_dir / "detection_summary.json").exists()
        assert (output_dir / "detection_results.csv").exists()
        assert (output_dir / "benchmark_result.json").exists()

        summary = json.loads(
            (output_dir / "detection_summary.json").read_text(encoding="utf-8")
        )
        assert summary["total_detections"] == 0

        # CSV はヘッダ行のみ (検出0件)
        csv_path = output_dir / "detection_results.csv"
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0


class TestWriteReportsNoAnnotation:
    """GT アノテーション不在時の write_reports テスト."""

    def test_skips_map_and_confusion_matrix(
        self,
        tmp_path: Path,
        sample_predictions: dict[str, list[Detection]],
    ) -> None:
        """annotation_path 未指定時は mAP 評価と混同行列をスキップすることを確認."""
        config: DetectionConfigDict = {
            "class_names": ["cat", "dog"],
        }
        ctx = _make_context(tmp_path, ["cat", "dog"])
        image_files = [Path(f"images/{name}") for name in sample_predictions]
        model_path = tmp_path / "model"
        model_path.mkdir()

        write_reports(config, image_files, sample_predictions, ctx, model_path)

        output_dir = ctx.saver.output_dir
        assert (output_dir / "detection_summary.json").exists()
        assert (output_dir / "detection_results.csv").exists()
        assert (output_dir / "benchmark_result.json").exists()
        # annotation_path 未指定 → 混同行列は生成されない
        assert not (output_dir / "confusion_matrix.html").exists()

        # benchmark_result に detection_metrics が含まれないことを確認
        result = json.loads(
            (output_dir / "benchmark_result.json").read_text(encoding="utf-8")
        )
        assert result["detection_metrics"] is None

    def test_nonexistent_annotation_path(
        self,
        tmp_path: Path,
        sample_predictions: dict[str, list[Detection]],
    ) -> None:
        """存在しない annotation_path を指定した場合もエラーにならないことを確認."""
        config: DetectionConfigDict = {
            "annotation_path": str(tmp_path / "nonexistent.json"),
            "class_names": ["cat", "dog"],
        }
        ctx = _make_context(tmp_path, ["cat", "dog"])
        image_files = [Path(f"images/{name}") for name in sample_predictions]
        model_path = tmp_path / "model"
        model_path.mkdir()

        write_reports(config, image_files, sample_predictions, ctx, model_path)

        output_dir = ctx.saver.output_dir
        assert (output_dir / "detection_summary.json").exists()
        assert not (output_dir / "confusion_matrix.html").exists()


class TestWriteReportsConfigSave:
    """config_path 指定時の設定ファイル保存テスト."""

    def test_saves_config_file(
        self,
        tmp_path: Path,
        sample_predictions: dict[str, list[Detection]],
    ) -> None:
        """config_path 指定時に設定ファイルがコピーされることを確認."""
        config: DetectionConfigDict = {
            "class_names": ["cat", "dog"],
        }
        # config_path 用のファイルを作成
        config_file = tmp_path / "my_config.py"
        config_file.write_text("# config", encoding="utf-8")

        ctx = _make_context(tmp_path, ["cat", "dog"])
        image_files = [Path(f"images/{name}") for name in sample_predictions]
        model_path = tmp_path / "model"
        model_path.mkdir()

        write_reports(
            config,
            image_files,
            sample_predictions,
            ctx,
            model_path,
            config_path=str(config_file),
        )

        output_dir = ctx.saver.output_dir
        assert (output_dir / "my_config.py").exists()
