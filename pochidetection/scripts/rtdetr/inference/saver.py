"""推論結果を保存するクラス."""

from pathlib import Path

from PIL import Image


class InferenceSaver:
    """推論結果を保存.

    Attributes:
        _output_dir: 出力ディレクトリ.
    """

    def __init__(self, base_dir: Path) -> None:
        """InferenceSaverを初期化.

        inference_001, inference_002 のようにインクリメントしたディレクトリを作成.

        Args:
            base_dir: 親ディレクトリのパス (model_path).
        """
        self._output_dir = self._create_numbered_dir(base_dir)

    def _create_numbered_dir(self, base_dir: Path) -> Path:
        """連番付きディレクトリを作成.

        Args:
            base_dir: 親ディレクトリ.

        Returns:
            作成したディレクトリのパス.
        """
        # 既存の inference_XXX ディレクトリを検索
        existing = list(base_dir.glob("inference_[0-9][0-9][0-9]"))
        if existing:
            # 最大番号を取得
            max_num = max(int(d.name.split("_")[1]) for d in existing)
            next_num = max_num + 1
        else:
            next_num = 1

        output_dir = base_dir / f"inference_{next_num:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def save(self, image: Image.Image, filename: str) -> Path:
        """画像を保存.

        Args:
            image: 保存する画像.
            filename: 元のファイル名.

        Returns:
            保存先のパス.
        """
        # ファイル名に _result を付加
        path = Path(filename)
        output_filename = f"{path.stem}_result{path.suffix}"
        output_path = self._output_dir / output_filename

        image.save(output_path)
        return output_path

    @property
    def output_dir(self) -> Path:
        """出力ディレクトリを取得.

        Returns:
            出力ディレクトリのパス.
        """
        return self._output_dir
