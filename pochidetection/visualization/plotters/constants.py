"""プロッター共通定数."""

LEGEND_CONFIG: dict[str, object] = {
    "yanchor": "top",
    "y": 1,
    "xanchor": "left",
    "x": 1.02,
}
"""凡例設定 (グラフ右上外)."""


_SIDE_BY_SIDE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        .container {{
            display: flex;
            justify-content: center;
            gap: 20px;
        }}
    </style>
</head>
<body>
    <h1 style="text-align: center;">{heading}</h1>
    <div class="container">
        {body}
    </div>
</body>
</html>
"""


def render_side_by_side_html(title: str, heading: str, body: str) -> str:
    """2 グラフ以上を横並び表示する共通 HTML を生成する.

    Args:
        title: <title> に出力するテキスト.
        heading: <h1> に出力するテキスト.
        body: .container 内に展開する HTML 断片 (各 <div> を連結済み).

    Returns:
        レンダリング済み HTML 文字列.
    """
    return _SIDE_BY_SIDE_TEMPLATE.format(title=title, heading=heading, body=body)
