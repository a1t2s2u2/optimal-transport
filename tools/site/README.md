# tools/site — TeX からセミナーサイトを生成する

LaTeX 原稿（`tex/`）を唯一の source of truth として、読むための Web サイトを生成する。
全セミナーで同じエンジンを使い、セミナー固有の情報だけを設定ファイルに切り出す。

```
seminar/<名前>/tex/{main,foundations}/*.tex   ← ★ ここだけを編集する
      │  tools/site/tex2md.mjs      （tex → markdown 中間表現）
      ▼
seminar/<名前>/site/content/*.md     ← 生成物（gitignore・手で触らない）
      │  tools/site/build.mjs       （markdown → html）
      ▼
seminar/<名前>/site/dist/            ← 生成物（gitignore）。デプロイ用サイト
  index.html / graph.html / main/*.html / appendix/*.html / styles.css / app.js / graph.js
```

## 使い方

```sh
node tools/site/tex2md.mjs seminar/cuturi [--strict]
node tools/site/build.mjs  seminar/cuturi
```

リポジトリ直下では `make cuturi-site` が両方を実行する。
`--strict` は lint 警告（未解決の `\ref`・数式の外に残った未変換マクロ）で失敗する。CI 用。

## 新しいセミナーを追加する

1. `seminar/<名前>/tex/` に `main.tex` / `preamble.tex` / `main/` / `foundations/` を置く
2. `seminar/<名前>/site.config.mjs` を書く（下記）
3. `Makefile` と `.github/workflows/pages.yml` にターゲットを足す

## site.config.mjs

必須は `title` と `chapters` だけ。残りはすべて既定値がある。

```js
export default {
  title: "計算最適輸送",              // ヘッダーと各ページの <title> に出る
  logo: "OT",                        // ヘッダー左のロゴ文字
  siteName: "計算最適輸送セミナー",   // ランディングの <title>（既定 "<title>セミナー"）
  landingTitle: "計算最適輸送",       // ランディングの大見出し（既定 title）
  landingSubtitle: "…",              // 大見出しの下（HTML 可）
  landingFooter: "…",                // ランディング末尾（HTML 可）
  appendixHeading: "付録：前提知識",
  appendixSubheading: "…",

  chapters: [
    {
      tex: "main/01_assignment.tex", // tex/ からの相対パス
      md: "01-assignment.md",        // site/content/ に出るファイル名（並び順も決める）
      id: "assignment",              // ページの識別子。URL は main/<id>.html
      group: "main",                 // "main" | "appendix"（既定 "main"）
      nav: "最適割当",               // ヘッダーのナビ表示
      eyebrow: "1. Assignment",      // 章見出しの上の小見出し（任意）
      title: "最適割当問題",         // 章タイトル
    },
  ],

  macroOverrides: { … },   // preamble.tex から自動抽出したマクロの上書き
  macroIgnore: ["foo"],    // 自動抽出から除きたいマクロ名
  glossary: { … },         // 本文の [term:表示|id] から引く用語集
  demos: { … },            // tex の \demohint{名前} から差し込む図
  features: { … },         // UI 機能の ON/OFF
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

`seminar/<名前>/site-assets/` に `styles.css` / `app.js` / `graph.js` を置くと、
そのファイルだけ共通アセットの代わりに使われる。

## 定理番号

サイトの見出しに出る「Def 2.1.3」は、`preamble.tex` の tcolorbox 設定
（全定理環境が `definition` のカウンタを共有し、`number within=section` で節ごとにリセット）
を再現して振っている。PDF と同じ番号になる。この規約から外れた preamble を使う場合は
`lib/tex2md.mjs` の `buildNumberMap()` を合わせること。

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

外部 CDN から MathJax・Mermaid・Cytoscape を読み込む。ネットワークのない環境では
数式やグラフは描画されない（ソース文字列は表示される）。
