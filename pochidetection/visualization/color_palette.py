"""カラーパレット管理クラス."""


class ColorPalette:
    """クラスIDに対応する色を提供.

    視認性の高いカラーパレットを使用し, クラスIDに応じて異なる色を返す.

    Attributes:
        _colors: 使用するカラーパレット.
    """

    # 視認性の高いデフォルトパレット (20色)
    DEFAULT_COLORS = [
        "#FF0000",  # red
        "#00FF00",  # lime
        "#0000FF",  # blue
        "#FFFF00",  # yellow
        "#FF00FF",  # magenta
        "#00FFFF",  # cyan
        "#FF8000",  # orange
        "#8000FF",  # purple
        "#00FF80",  # spring green
        "#FF0080",  # rose
        "#80FF00",  # chartreuse
        "#0080FF",  # azure
        "#FF8080",  # light coral
        "#80FF80",  # light green
        "#8080FF",  # light blue
        "#FFFF80",  # light yellow
        "#FF80FF",  # light magenta
        "#80FFFF",  # light cyan
        "#C00000",  # dark red
        "#00C000",  # dark green
    ]

    def __init__(self, colors: list[str] | None = None) -> None:
        """ColorPaletteを初期化.

        Args:
            colors: カスタムカラーリスト. Noneの場合はデフォルトパレットを使用.
        """
        self._colors = colors if colors is not None else self.DEFAULT_COLORS

    def get_color(self, class_id: int) -> str:
        """クラスIDに対応する色を取得.

        Args:
            class_id: クラスID (0, 1, 2, ...).

        Returns:
            HEX形式の色コード (例: "#FF0000").
        """
        return self._colors[class_id % len(self._colors)]

    @property
    def colors(self) -> list[str]:
        """カラーパレットを取得.

        Returns:
            カラーパレットのリスト.
        """
        return self._colors
