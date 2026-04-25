"""prepare-demo サブコマンドが parser に登録されていることを検証."""

import pytest

from pochidetection.cli.parser import _create_parser


def test_prepare_demo_subcommand_registered() -> None:
    """parse_args で prepare-demo が認識され, 既定の input-size が 640x640."""
    parser = _create_parser()
    args = parser.parse_args(["prepare-demo"])
    assert args.command == "prepare-demo"
    assert args.input_size == [640, 640]


def test_prepare_demo_accepts_input_size() -> None:
    """--input-size HEIGHT WIDTH を受け取れること."""
    parser = _create_parser()
    args = parser.parse_args(["prepare-demo", "--input-size", "800", "1280"])
    assert args.input_size == [800, 1280]


def test_prepare_demo_rejects_single_input_size() -> None:
    """--input-size に値 1 つだけ渡すと SystemExit (nargs=2)."""
    parser = _create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["prepare-demo", "--input-size", "640"])
