"""ラベルマッピングクラス."""


class LabelMapper:
    """クラスIDをクラス名に変換.

    Attributes:
        _class_names: クラス名のリスト.
    """

    def __init__(self, class_names: list[str] | None = None) -> None:
        """LabelMapperを初期化.

        Args:
            class_names: クラス名のリスト. インデックスがクラスIDに対応.
                         Noneの場合は整数のラベルをそのまま文字列として返す.
        """
        self._class_names = class_names

    def get_label(self, class_id: int) -> str:
        """クラスIDに対応するラベル名を取得.

        Args:
            class_id: クラスID (0, 1, 2, ...).

        Returns:
            クラス名. class_namesが未設定またはIDが範囲外の場合は整数を文字列化.
        """
        if self._class_names is not None and 0 <= class_id < len(self._class_names):
            return self._class_names[class_id]
        return str(class_id)

    @property
    def class_names(self) -> list[str] | None:
        """クラス名リストを取得.

        Returns:
            クラス名のリスト. 未設定の場合はNone.
        """
        return self._class_names

    @property
    def num_classes(self) -> int | None:
        """クラス数を取得.

        Returns:
            クラス数. class_namesが未設定の場合はNone.
        """
        if self._class_names is None:
            return None
        return len(self._class_names)
