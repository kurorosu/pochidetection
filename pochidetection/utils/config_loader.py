"""設定ファイルの読み込みとバリデーション.

Python 設定ファイル (.py) を読み込み, Pydantic でバリデーションする.
"""

import importlib.util
import inspect
import re
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from pydantic import ValidationError

from pochidetection.configs.schemas import (
    DetectionConfigAdapter,
    DetectionConfigDict,
    normalize_architecture,
)

_BASE_CONFIG_NAME = "_base.py"


class ConfigLoader:
    """設定ファイルの読み込みとバリデーション.

    Python設定ファイルの個別変数を辞書として読み込み,
    Pydantic スキーマでバリデーションを行う.

    ベース設定ファイル (_base.py) が同一ディレクトリに存在する場合,
    ベースを先に読み込んでから個別設定で上書きマージする.

    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合.
        ValidationError: 設定値がスキーマと一致しない場合.
    """

    CONFIG_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

    @classmethod
    def load(cls, config_path: str | Path) -> DetectionConfigDict:
        """設定を読み込み, Pydantic バリデーションを行う.

        Args:
            config_path: 設定ファイルのパス.

        Returns:
            バリデーション済みの設定辞書.

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合.
            ValidationError: 設定値がスキーマと一致しない場合.

        Examples:
            >>> config = ConfigLoader.load("configs/rtdetr_coco.py")
            >>> config["num_classes"]
            4
            >>> config["model_name"]
            'PekingU/rtdetr_r50vd_coco_o365'
        """
        raw_config = cls._load_file(config_path)
        if isinstance(raw_config.get("architecture"), str):
            raw_config["architecture"] = normalize_architecture(
                raw_config["architecture"]
            )
        try:
            validated = DetectionConfigAdapter.validate_python(raw_config)
        except ValidationError as error:
            error.add_note(f"設定ファイルのバリデーションに失敗しました: {config_path}")
            raise
        return cast(DetectionConfigDict, validated.model_dump())

    @classmethod
    def _load_file(cls, config_path: str | Path) -> dict[str, Any]:
        """Python設定ファイルを読み込み辞書として返す.

        同一ディレクトリに _base.py が存在する場合,
        ベース設定を先に読み込み, 個別設定で上書きマージする.

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

        config: dict[str, Any] = {}

        # ベース設定があればマージ
        base_path = path.parent / _BASE_CONFIG_NAME
        if base_path.exists() and path.name != _BASE_CONFIG_NAME:
            config.update(cls._extract_config(base_path))

        config.update(cls._extract_config(path))
        return config

    @staticmethod
    def write_config(config: DetectionConfigDict, path: Path) -> None:
        """設定辞書を Python ファイルとして書き出す.

        ベースマージ済みの全パラメータを単一ファイルに展開して保存する.

        Args:
            config: 設定辞書.
            path: 書き出し先のパス.
        """
        lines = [
            '"""Saved config (merged)."""',
            "",
        ]
        for key, value in sorted(config.items()):
            lines.append(f"{key} = {value!r}")
        lines.append("")  # trailing newline
        path.write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def _extract_config(cls, path: Path) -> dict[str, Any]:
        """Python ファイルからモジュール属性を設定辞書として抽出する.

        Args:
            path: Python 設定ファイルのパス.

        Returns:
            設定辞書.

        Raises:
            FileNotFoundError: ファイルを読み込めない場合.

        Notes:
            ローカル Python config ファイルを
            ``importlib.util.spec_from_file_location`` と
            ``spec.loader.exec_module()`` で読み込む. これは
            ``path`` 先の Python コードを現プロセス内で実行することと
            等価であり, モジュール本体の式や副作用 (import 文,
            ファイル I/O, 関数定義, グローバル変数の初期化など) は
            すべて評価される. 本メソッドは評価後のモジュール属性から
            設定辞書を抽出しているに過ぎない.

        Warning:
            本メソッドは自前で管理している trusted な config ファイル
            (``configs/*.py``) のみを対象とする. 外部入力
            (ユーザアップロードファイル, HTTP リクエストボディ,
            未検証のパス文字列, ネットワーク経由で取得した .py など) を
            ``path`` に渡してはいけない. ``exec_module()`` は任意
            コード実行と等価であり, ``python config.py`` を直接
            実行するのと同じセキュリティ境界とみなす必要がある.
        """
        spec = importlib.util.spec_from_file_location("config", path)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"設定ファイルを読み込めません: {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return cls._module_to_dict(module)

    @classmethod
    def _module_to_dict(cls, module: ModuleType) -> dict[str, Any]:
        """モジュール属性から設定値のみを辞書として抽出する.

        Args:
            module: Python モジュール.

        Returns:
            設定辞書.
        """
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
