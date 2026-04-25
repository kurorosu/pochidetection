# Changelog Layout

このディレクトリは最新でなくなった changelog を保持.
`CHANGELOG.md` は最新の changelog を保持し, 旧履歴を本ディレクトリへ移動.

## Files

- `0.17.x.md`: v0.17.0 の変更履歴.
- `0.16.x.md`: v0.16.0 の変更履歴.
- `0.15.x.md`: v0.15.0 の変更履歴.
- `0.14.x.md`: v0.14.0 ~ v0.14.1 の変更履歴.
- `0.13.x.md`: v0.13.0 の変更履歴.
- `0.12.x.md`: v0.12.0 ~ v0.12.1 の変更履歴.
- `0.11.x.md`: v0.11.0 の変更履歴.
- `0.10.x.md`: v0.10.0 の変更履歴.
- `0.9.x.md`: v0.9.0 の変更履歴.
- `0.8.x.md`: v0.8.0 の変更履歴.
- `0.7.x.md`: v0.7.0 の変更履歴.
- `0.6.x.md`: v0.6.0 ~ v0.6.4 の変更履歴.
- `0.5.x.md`: v0.5.0 の変更履歴.
- `0.4.x.md`: v0.4.0 の変更履歴.
- `0.3.x.md`: v0.3.0 の変更履歴.
- `0.2.x.md`: v0.2.0 の変更履歴.
- `0.1.x.md`: v0.1.0 の変更履歴.

## Update Flow

1. 新機能, 修正, リファクタは `CHANGELOG.md` の `[Unreleased]` に追記する.
2. PR本文を `tmp_PR.md` に作成するタイミングで, `[Unreleased]` も同時に更新する.
3. リリース時は `CHANGELOG.md` に新しいバージョン節を追加する.
4. 最新ではなくなった履歴を `changelogs/<major>.<minor>.x.md` へ移動する.
5. 各バージョンは `Added`, `Changed`, `Fixed`, `Removed` の順で記載する.
6. `CHANGELOG.md` の `Archived Changelogs` を更新する.
7. PR参照は `([#123](https://github.com/kurorosu/pochidetection/pull/123))` の明示リンク形式で記載する.
8. PR参照には **PR 番号** を使う (Issue 番号ではない). PR 作成前は `N/A.` とする.

## Notes

- `[Unreleased]` の未確定項目は `N/A.` を使う.
- リリース済みバージョンで該当項目がない場合は `なし.` を使う.
- 1行要約より, 利用者が判断できる粒度の変更内容を優先する.
