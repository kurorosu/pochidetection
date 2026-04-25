"""serve サブコマンドが parser に登録されていることを検証."""

from pochidetection.cli.parser import _create_parser


def test_serve_subcommand_registered() -> None:
    """parse_args で serve が認識されること."""
    parser = _create_parser()
    args = parser.parse_args(["serve", "-m", "work_dirs/dummy/best"])
    assert args.command == "serve"
    assert args.model_path == "work_dirs/dummy/best"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.config is None


def test_serve_allows_omitting_model_path() -> None:
    """-m 省略時は model_path が None (pretrained 経路)."""
    parser = _create_parser()
    args = parser.parse_args(["serve"])
    assert args.command == "serve"
    assert args.model_path is None


def test_serve_accepts_host_port_config() -> None:
    """--host / --port / -c を受け取れること."""
    parser = _create_parser()
    args = parser.parse_args(
        [
            "serve",
            "-m",
            "best",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "-c",
            "configs/rtdetr_coco.py",
        ]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.config == "configs/rtdetr_coco.py"
