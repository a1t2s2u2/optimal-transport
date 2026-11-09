# tools/notation

セミナーの tex について、記法と文章構成の規則を検査する。証明の正しさは検査しない。

```sh
node tools/notation/lint.mjs <セミナーのディレクトリ>   # 例: make wasserstein-lint
```

違反を `ファイル:行: [規則] 内容` の形で出し、1 件でもあれば終了コード 1 で終わる。

## 規則

| 規則 | 内容 |
|---|---|
| `forbidden` | `site.config.mjs` の `notation.forbidden` に挙げた書き方を使っていない |
| `macro-literal` | `preamble.tex` のマクロと同じものを直書きしていない（`\mathbb{R}` ではなく `\R`） |
| `definition-term` | 定義ブロックに定義される語（`\textbf{…}`）がある |
| `proof-or-source` | 定理・補題・命題・系に証明がある。後の節で証明するときは `\begin{proof}[定理~\ref{…}の証明]`、証明しないときは「（証明は省略する．〈出典〉）」と書く |

参照語と参照先の種類の一致（`定理~\ref{lem:…}` の検出）は `tools/site` の `--strict` が検査する。

## セミナー固有の禁止記法

`seminar/<名前>/site.config.mjs` の `notation.forbidden` に、正規表現と代わりの書き方を書く。

```js
notation: {
  forbidden: [
    { pattern: /\\phi(?![a-zA-Z])/, use: "\\varphi", why: "φ の字形を統一する" },
  ],
},
```

## 検査しない行

記法そのものに言及する行（「文献では $T_\#\mu$ とも書く」など）には、行末に
`% notation: ignore` を書く。ブロックの規則では、`\begin` の行に書くとそのブロックを検査しない。
