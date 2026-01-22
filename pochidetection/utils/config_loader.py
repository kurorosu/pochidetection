"""設定ファイルの読み込みとバリデーション.

Python設定ファイル (.py) を読み込み, バリデーションを行う.
pochisegmentationと同じ形式: 個別変数を辞書に変換.
"""

import importlib.util
from pathlib import Path
from typing import Any


class ConfigLoader:
    """設定ファイルの読み込みとバリデーション.

    Python設定ファイルの個別変数を辞書として読み込み,
    バリデーションとデフォルト値の適用を行う.

    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合.
        KeyError: 必須キーがない場合.
        ValueError: 設定値が許可リストにない場合.
        TypeError: 設定値の型が不正な場合.
    """

    REQUIRED_KEYS: list[str] = ["data_root", "num_classes"]

    ALLOWED_VALUES: dict[str, list[Any]] = {
        "architecture": ["RTDetr"],
        "loss": ["DetectionLoss"],
        "device": ["cuda", "cpu"],
    }

    TYPE_CHECKS: dict[str, type | tuple[type, ...]] = {
        "num_classes": int,
        "batch_size": int,
        "epochs": int,
        "learning_rate": (int, float),
        "image_size": int,
        "pretrained": bool,
    }

    DEFAULTS: dict[str, Any] = {
        "architecture": "RTDetr",
        "model_name": "PekingU/rtdetr_r50vd",
        "pretrained": True,
        "image_size": 640,
        "batch_size": 4,
        "epochs": 100,
        "learning_rate": 1e-4,
        "loss": "DetectionLoss",
        "metrics": "DetectionMetrics",
        "dataset": "CocoDetectionDataset",
        "device": "cuda",
        "work_dir": "work_dirs",
    }

    @classmethod
    def load(cls, config_path: str | Path) -> dict[str, Any]:
        """設定を読み込み, バリデーションとデフォルト値適用.

        Args:
            config_path: 設定ファイルのパス.

        Returns:
            バリデーション済みの設定辞書.

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合.
            KeyError: 必須キーがない場合.
            ValueError: 設定値が許可リストにない場合.
            TypeError: 設定値の型が不正な場合.
        """
        config = cls._load_file(config_path)
        cls._validate_required(config)
        cls._validate_values(config)
        cls._validate_types(config)
        return cls._apply_defaults(config)

    @classmethod
    def _load_file(cls, config_path: str | Path) -> dict[str, Any]:
        """Python設定ファイルを読み込み辞書として返す.

        Args:
            config_path: 設定ファイルのパス.

        Returns:
            設定辞書.

        Raises:
            FileNotFoundError: ファイルが見つからない場合.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

        spec = importlib.util.spec_from_file_location("config", path)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"設定ファイルを読み込めません: {config_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # モジュールの公開属性を辞書として取得 (アンダースコアで始まらないもの)
        config: dict[str, Any] = {}
        for key in dir(module):
            if not key.startswith("_"):
                config[key] = getattr(module, key)

        return config

    @classmethod
    def _validate_required(cls, config: dict[str, Any]) -> None:
        """必須項目チェック.

        Args:
            config: 設定辞書.

        Raises:
            KeyError: 必須キーがない場合.
        """
        missing = [key for key in cls.REQUIRED_KEYS if key not in config]
        if missing:
            raise KeyError(f"必須キーが設定に存在しません: {missing}")

    @classmethod
    def _validate_values(cls, config: dict[str, Any]) -> None:
        """許可値チェック.

        Args:
            config: 設定辞書.

        Raises:
            ValueError: 設定値が許可リストにない場合.
        """
        for key, allowed in cls.ALLOWED_VALUES.items():
            if key in config and config[key] not in allowed:
                raise ValueError(
                    f"設定値が不正です: {key}={config[key]}. 許可値: {allowed}"
                )

    @classmethod
    def _validate_types(cls, config: dict[str, Any]) -> None:
        """型チェック.

        Args:
            config: 設定辞書.

        Raises:
            TypeError: 設定値の型が不正な場合.
        """
        for key, expected_type in cls.TYPE_CHECKS.items():
            if key in config and not isinstance(config[key], expected_type):
                raise TypeError(
                    f"設定値の型が不正です: {key}={config[key]} "
                    f"(期待: {expected_type}, 実際: {type(config[key])})"
                )

    @classmethod
    def _apply_defaults(cls, config: dict[str, Any]) -> dict[str, Any]:
        """デフォルト値を適用.

        Args:
            config: 設定辞書.

        Returns:
            デフォルト値が適用された設定辞書.
        """
        result = cls.DEFAULTS.copy()
        result.update(config)
        return result
