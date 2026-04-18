## Summary

- `pochidetection/interfaces/pipeline.py` の `Union` import 削除
- `ImageInput` エイリアスを `X | Y` 記法 (`TypeAlias` 付き) に書き換え
- 他ファイルに古い型注釈 (`typing.Union` / `Optional` / `Dict` / `List` / `Tuple`) が残っていないことを grep で確認

## Related Issue

Closes #461

## Changes

- `pochidetection/interfaces/pipeline.py`: `Union` 除去, `ImageInput: TypeAlias = Image.Image | np.ndarray` に変更

## Test Plan

- [x] `uv run mypy pochidetection/interfaces/pipeline.py` で pipeline.py に新規エラーが発生しないことを確認
- [x] `uv run pytest tests/test_api/test_backends.py` が pass (`ImageInput` を import している側のテスト)
- [x] `grep` で他ファイルに `typing.Union` / `typing.Optional` / `typing.Dict` / `typing.List` / `typing.Tuple` が残っていないことを確認
- [x] `uv run pre-commit run --all-files` のうち black / isort / pydocstyle / mypy / secrets は全 pass (pytest は worktree 環境に tensorrt 未インストールのため skip. `.venv` の junction が sandbox 制約で作成不可だったため新規 `.venv` が生成され, 既定リポの手動インストール済み tensorrt が取り込めない)

## Checklist

- [x] `uv run pre-commit run --all-files` の tensorrt 非依存部 (black / isort / mypy / pydocstyle / secrets / 改行系) 全 pass
