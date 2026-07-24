# tools/site — TeX から読むための Web サイトを生成する

LaTeX 原稿（`tex/`）を唯一の source of truth として、読むための Web サイトを生成する。
エンジンは文書に依存しない。文書ごとの違いは `site.config.mjs` に閉じ込める。

```
<文書のディレクトリ>/
  tex/{main,foundations}/*.tex   ← ★ ここだけを編集する
        │  tools/site/tex2md.mjs      （tex → markdown 中間表現）
        ▼
  site/content/*.md               ← 生成物（gitignore・手で触らない）
        │  tools/site/build.mjs       （markdown → html）
        ▼
  site/dist/                      ← 生成物（gitignore）。デプロイ用サイト
    index.html / graph.html / main/*.html / appendix/*.html
    styles.css / app.js / graph.js
```

## 使い方

引数は「`tex/` と `site.config.mjs` を含むディレクトリ」。

```sh
node tools/site/tex2md.mjs <文書のディレクトリ> [--strict]
node tools/site/build.mjs  <文書のディレクトリ>
```

文書がリポジトリ直下にあるなら `.`、複数の文書を並べているなら `seminar/cuturi` のように渡す。
`--strict` は lint 警告（未解決の `\ref`・数式の外に残った未変換マクロ）で失敗する。CI 用。

## 文書を追加する

1. `<ディレクトリ>/tex/` に `main.tex` / `preamble.tex` / `main/` / `foundations/` を置く
2. `<ディレクトリ>/site.config.mjs` を書く（下記）
3. `Makefile` と GitHub Actions のワークフローにターゲットを足す

## site.config.mjs

必須は `title` と `chapters` だけ。残りはすべて既定値がある。

```js
export default {
  title: "ノートのタイトル",       // ヘッダーと各ページの <title> に出る
  logo: "N",                       // ヘッダー左のロゴ文字（既定 "OT"）
  siteName: "…",                   // ランディングの <title>（既定 "<title>セミナー"）
  landingTitle: "…",               // ランディングの大見出し（既定 title）
  landingSubtitle: "…",            // 大見出しの下（HTML 可）
  landingFooter: "…",              // ランディング末尾（HTML 可）
  appendixHeading: "付録：前提知識",
  appendixSubheading: "…",
  lang: "ja",

  chapters: [
    {
      tex: "main/01_introduction.tex", // tex/ からの相対パス
      md: "01-introduction.md",        // site/content/ に出る名前（並び順もこれで決まる）
      id: "introduction",              // ページの識別子。URL は main/<id>.html
      group: "main",                   // "main" | "appendix"（既定 "main"）
      nav: "距離空間",                 // ヘッダーのナビ表示
      eyebrow: "1. Metric Spaces",     // 章見出しの上の小見出し（任意）
      title: "距離空間と収束",         // 章タイトル
    },
  ],

  macroOverrides: { … },   // preamble.tex から自動抽出したマクロの上書き
  macroIgnore: ["foo"],    // 自動抽出から除きたいマクロ名
  glossary: { … },         // 本文の [term:表示|id] から引く用語集
  demos: { … },            // tex の \demohint{名前} から差し込む図
  features: { … },         // UI 機能の ON/OFF

  // 変換規則の上書き（通常は不要）
  texDir: "tex",           // 原稿のディレクトリ
  texSubdirs: ["main", "foundations"],
  blockEnvs: { … },        // tex の環境名 → [コンテナ, 見出し接頭辞]
  graphTypes: [ … ],       // 依存グラフに載せる種別
};
```

### 数式マクロ

MathJax に渡すマクロは `tex/preamble.tex` の `\newcommand` / `\renewcommand` /
`\DeclareMathOperator` から自動抽出する。**設定に書き写す必要はない**。

MathJax が解釈できない綴りだけ `macroOverrides` で差し替える。例えば stmaryrd の
`\llbracket` は MathJax にないので、

```js
macroOverrides: {
  range: ["{\\lbrack\\!\\lbrack}#1{\\rbrack\\!\\rbrack}", 1],
},
```

### features

すべて既定 `true`。

| キー | 内容 |
| --- | --- |
| `chapterStats` | ランディングの章カードに「46 定義・7 定理」を出す |
| `heroDecoration` | ランディング見出し背景の装飾 SVG |
| `tocProgress` | 章内目次の現在位置インジケータ |
| `fadeIn` | ブロックのスクロール・フェードイン |
| `refPulse` | 参照クリック時に本文側のブロックを光らせる |
| `keyboardHelp` | `?` でショートカット一覧を表示 |

### 見た目を差し替える

`<文書のディレクトリ>/site-assets/` に `styles.css` / `app.js` / `graph.js` を置くと、
そのファイルだけ共通アセットの代わりに使われる。

## 対応している tex の書き方

| tex | サイト |
| --- | --- |
| `\begin{definition}{タイトル}{ラベル}` など | 種別ごとに色分けしたブロック |
| `\begin{proof}`（定理環境の直後） | 折りたたみの「証明」 |
| `\begin{memo*}` | 番号なしの補足 |
| `\begin{algorithm}{ラベル}` | アルゴリズムのブロック（行末 `\\` が改行） |
| `\ref{def:foo}` | 「定義 1.1.1」のボタン。クリックで本文を表示 |
| `\section` / `\subsection` / `\subsubsection` | `h2` / `h3` / 太字 |
| `itemize` / `enumerate`（`\item[…]` も可） | リスト |
| `$…$` / `\[…\]` / `align*` | MathJax |
| `\textbf` / `\textit` / `\emph` / `\paragraph` | 強調・見出し |
| `figure` / `tikzpicture` / `center` | **省略される**（PDF にのみ出る） |
| `\demohint{名前}` | `demos` に登録した HTML を差し込む |

`\cite` / `\footnote` / `\verb` は解釈しない。使うと lint 警告になる（`--strict` では失敗）。

## 定理番号

見出しの「Def 2.1.3」は `preamble.tex` の tcolorbox 設定
（全定理環境が `definition` のカウンタを共有し、`number within=section` で節ごとにリセット）
を再現して振っている。PDF と同じ番号になる。この規約から外れた preamble を使う場合は
`lib/tex2md.mjs` の `buildNumberMap()` も合わせること。

## 構成

| ファイル | 役割 |
| --- | --- |
| `tex2md.mjs` | CLI。tex → content/*.md |
| `build.mjs` | CLI。content/*.md → dist/ |
| `lib/config.mjs` | site.config.mjs の読み込みと既定値 |
| `lib/macros.mjs` | preamble.tex → MathJax マクロ表 |
| `lib/tex2md.mjs` | TeX パーサと markdown 出力 |
| `lib/markdown.mjs` | 中間 markdown → HTML |
| `lib/blocks.mjs` | ブロック種別（ラベル・配色・id 接頭辞）の定義元 |
| `lib/templates.mjs` | ランディング・章・依存グラフの HTML |
| `lib/graph.mjs` | 参照関係から依存グラフのデータを組む |
| `assets/` | dist へコピーする styles.css / app.js / graph.js |

Node.js 22 以上で動く。依存パッケージはない。
外部 CDN から MathJax・Mermaid・Cytoscape を読み込むため、ネットワークのない環境では
数式やグラフは描画されない（ソース文字列は表示される）。
