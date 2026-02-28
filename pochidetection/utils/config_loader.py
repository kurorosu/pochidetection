"""設定ファイルの読み込みとバリデーション.

Python 設定ファイル (.py) を読み込み, Pydantic でバリデーションする.
"""

import importlib.util
import inspect
import re
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from pochidetection.configs.schemas import DetectionConfig


class ConfigLoader:
    """設定ファイルの読み込みとバリデーション.

    Python設定ファイルの個別変数を辞書として読み込み,
    Pydantic スキーマでバリデーションを行う.

    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合.
        ValidationError: 設定値がスキーマと一致しない場合.
    """

    CONFIG_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

    @classmethod
    def load(cls, config_path: str | Path) -> dict[str, Any]:
        """設定を読み込み, Pydantic バリデーションを行う.

        Args:
            config_path: 設定ファイルのパス.

        Returns:
            バリデーション済みの設定辞書.

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合.
            ValidationError: 設定値がスキーマと一致しない場合.
        """
        raw_config = cls._load_file(config_path)
        try:
            validated = DetectionConfig.model_validate(raw_config)
        except ValidationError as error:
            error.add_note(f"設定ファイルのバリデーションに失敗しました: {config_path}")
            raise
        return cast(dict[str, Any], validated.model_dump())

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

        # モジュール属性のうち, 設定値として扱うキーのみ抽出する.
        config: dict[str, Any] = {}
        for key in dir(module):
            if not cls.CONFIG_KEY_PATTERN.match(key):
                continue

            value = getattr(module, key)
            if (
                inspect.ismodule(value)
                or inspect.isclass(value)
                or inspect.isroutine(value)
            ):
                continue
            config[key] = value

        return config
