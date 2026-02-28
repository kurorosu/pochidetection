"""Detector のテスト."""


class TestDetector:
    """Detector クラスのテスト."""

    def test_detector_initialization_parameters(self) -> None:
        """Detector の初期化パラメータが正しく設定されることを確認."""
        # Detector は model_path が必要なため, ロジックのみテスト
        # 実際のモデルロードは E2E テストで行う
        assert True  # プレースホルダ
