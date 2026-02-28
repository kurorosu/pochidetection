"""Detector のテスト."""

from pochidetection.scripts.rtdetr.inference.detection import Detection


class TestDetection:
    """Detection dataclass のテスト."""

    def test_detection_stores_values(self) -> None:
        """Detection が値を正しく保持することを確認."""
        detection = Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.95, label=1)

        assert detection.box == [10.0, 20.0, 30.0, 40.0]
        assert detection.score == 0.95
        assert detection.label == 1

    def test_detection_equality(self) -> None:
        """同じ値の Detection が等しいことを確認."""
        d1 = Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.95, label=1)
        d2 = Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.95, label=1)

        assert d1 == d2

    def test_detection_inequality(self) -> None:
        """異なる値の Detection が等しくないことを確認."""
        d1 = Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.95, label=1)
        d2 = Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.80, label=1)

        assert d1 != d2

    def test_detection_box_format(self) -> None:
        """Detection の box が [x1, y1, x2, y2] 形式であることを確認."""
        detection = Detection(box=[0.0, 0.0, 100.0, 200.0], score=0.5, label=0)

        assert len(detection.box) == 4
        x1, y1, x2, y2 = detection.box
        assert x2 > x1
        assert y2 > y1
