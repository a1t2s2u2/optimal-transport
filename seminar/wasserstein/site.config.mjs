// Wasserstein 距離セミナーのサイト設定。
// 変換エンジン本体は tools/site/。ここにはこのセミナー固有の情報だけを置く。
// 数式マクロは tex/preamble.tex から自動抽出されるので、ここに書き写す必要はない。

import { gluingDiagram } from "./site-demos.mjs";

export default {
  // --- 表示 ---
  title: "Wasserstein 距離",
  logo: "OT",
  siteName: "Wasserstein 距離セミナー",
  landingTitle: "Wasserstein 距離",
  landingSubtitle: "定義から距離性の証明，測度分解定理まで",
  landingFooter: "参考文献: Villani (2009), 桑江ほか (2015)",
  appendixHeading: "付録：前提知識",
  appendixSubheading: "発表では省略した数学的前提を網羅した完全版．本編から参照される．",

  // --- 章立て（tex は tex/ からの相対パス、md は site/content/ に出る）---
  chapters: [
    {
      tex: "main/00_introduction.tex",
      md: "00-introduction.md",
      id: "introduction",
      group: "main",
      nav: "導入",
      eyebrow: "1. Introduction",
      title: "導入",
    },
    {
      tex: "main/01_wasserstein_metrics.tex",
      md: "01-wasserstein-metrics.md",
      id: "wasserstein-metrics",
      group: "main",
      nav: "Wₚ の距離性",
      eyebrow: "2. The Metric Wₚ",
      title: "Wasserstein 距離 Wₚ",
    },
    {
      tex: "main/02_disintegration.tex",
      md: "02-disintegration.md",
      id: "disintegration",
      group: "main",
      nav: "測度分解定理",
      eyebrow: "3. Disintegration",
      title: "測度分解定理",
    },
    {
      tex: "foundations/00_preliminaries.tex",
      md: "A0-preliminaries.md",
      id: "found-preliminaries",
      group: "appendix",
      nav: "距離空間と測度",
      eyebrow: "付録 A. Metric Spaces & Measures",
      title: "距離空間と測度の準備",
    },
  ],

  demos: {
    gluing: gluingDiagram,
  },

  // --- 記法の検査（tools/notation）。禁止する書き方と、その代わり ---
  notation: {
    forbidden: [
      { pattern: /\\phi(?![a-zA-Z])/, use: "\\varphi", why: "φ の字形を統一する" },
      { pattern: /\\epsilon(?![a-zA-Z])/, use: "\\varepsilon", why: "ε の字形を統一する" },
      {
        pattern: /\\su[bp]seteq(?![a-zA-Z])/,
        use: "\\subset / \\supset",
        why: "Villani に合わせ ⊂ を等号込みで使う（Villani (2009) p.44 Lemma 4.4）",
      },
      { pattern: /_\{?\\(?:sharp|#)/, use: "\\mu\\circ T^{-1}", why: "押し出し記法は使わず像測度で書く" },
      { pattern: /\\mathrm\{pr\}/, use: "周辺分布・座標成分の直接の記述", why: "座標射影を記号化しない" },
      {
        pattern: /_\{?[A-Za-z0-9]+\}?\s*\\Rightarrow/,
        use: "\\weakto",
        why: "\\Rightarrow は含意専用．弱収束には使わない",
      },
      { pattern: /[、。]/, use: "，．", why: "句読点を統一する" },
    ],
  },
};
