## リポジトリ概要

最適輸送のセミナー資料（tex / site）と関連ツールを管理するリポジトリ。

- `seminar/cuturi/` — Peyré–Cuturi _Computational Optimal Transport_ に基づく資料（詳細は `seminar/cuturi/CLAUDE.md`）
- `seminar/wasserstein/` — Wasserstein 距離の理論と性質（詳細は `seminar/wasserstein/CLAUDE.md`）

## セミナー方針（共通）

- 未定義の用語や概念は、必ず定義し、出典を明示する
- 一般的ではない用語や記法は勝手に使わない
- 記法は現代的な標準（Villani / Peyré–Cuturi）に従う
- 数学書の記述スタイルを遵守する

## ビルド

- tex を変更したら、対応するサイトのビルドまで実施する（`make cuturi-site` / `make wasserstein-site`）

## Git 運用

- 発表日 `feat/MMDD` ブランチを作成し、`main` にマージする。適宜 `feat/MMDD` からブランチを切って作業する
- コミットメッセージは `prefix: 日本語` 形式（例 `fix: …`, `docs: …`）
- コミット / PR に co-authored-by や ClaudeCode は記載しない
