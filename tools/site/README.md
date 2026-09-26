# tools/site

LaTeX 原稿を、定理と相互参照を読みやすくした静的サイトへ変換する。
原稿は `tex/` だけを編集し、`site/` 以下はすべて再生成する。

```text
tex/*.tex
  └─ tex2md.mjs → site/content/*.md
                       └─ build.mjs → site/dist/*.html
```

## 使い方

```sh
node tools/site/tex2md.mjs <文書ディレクトリ> [--strict]
node tools/site/build.mjs  <文書ディレクトリ>
```

`--strict` は未解決の `\ref` や未変換マクロがあると失敗する。CIでは必ず使う。

文書ディレクトリには `tex/` と `site.config.mjs` を置く。設定の必須項目は
`title` と `chapters` だけである。

```js
export default {
  title: "ノートのタイトル",
  chapters: [
    {
      tex: "main/01_intro.tex",
      md: "01-intro.md",
      id: "intro",
      nav: "導入",
      title: "導入",
      group: "main",       // または "appendix"
      eyebrow: "1. Intro", // 任意
    },
  ],
  macroOverrides: {},
  demos: {},
};
```

## 対応する原稿

- `\section`、`\subsection`、`\subsubsection`
- `definition`、`theorem`、`proposition`、`lemma`、`claim`、`corollary`
- `remark`、`example`、`proof`
- `itemize`、`enumerate`
- `$…$`、`\[…\]`、`align*`
- `\textbf`、`\textit`、`\emph`、`\paragraph`
- `\ref` による章横断参照
- `\demohint{名前}` による文書固有のHTML図

`figure`、`tikzpicture`、`center` はサイトでは省略する。引用、脚注、verbなど
未対応の命令はlintで検出する。

## 設計

- PDFと同じ定理番号を再現する。
- 参照をクリックすると、参照先の定義や定理をその場で表示する。
- MathJaxのマクロは `tex/preamble.tex` から抽出する。
- 文書固有の章立てとデモだけを `site.config.mjs` に置く。
- Node.js 22以外の実行時依存を持たない。

MathJaxとGoogle FontsはCDNから読み込む。ネットワークがない場合も本文は読めるが、
数式はソース表記になる。

## ファイル

```text
tex2md.mjs          CLI: tex → 中間markdown
build.mjs           CLI: markdown → HTML
lib/tex2md.mjs      TeXの限定パーサ、番号・参照・lint
lib/markdown.mjs    中間markdownのHTML化
lib/config.mjs      設定の検証と既定値
lib/macros.mjs      MathJaxマクロ抽出
lib/templates.mjs   ランディング・章ページ
assets/             共通CSSと参照プレビューJS
```
