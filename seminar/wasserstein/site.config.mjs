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
  landingSubtitle: "定義から距離性の証明まで",
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
      eyebrow: "0. Introduction",
      title: "導入",
    },
    {
      tex: "main/01_wasserstein_metrics.tex",
      md: "01-wasserstein-metrics.md",
      id: "wasserstein-metrics",
      group: "main",
      nav: "Wₚ の距離性",
      eyebrow: "1. The Metric Wₚ",
      title: "Wasserstein 距離 Wₚ",
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

  // --- 用語集（本文の [term:表示|id] から引く）---
  glossary: {
    polish: {
      title: "Polish 空間",
      body: "完備かつ可分な距離空間。確率測度の tightness や正則条件付き確率を扱う標準的な設定。",
    },
    coupling: {
      title: "結合",
      body: "二つの確率測度を周辺分布にもつ積空間上の確率測度。輸送計画とも呼ぶ。",
    },
  },

  demos: {
    gluing: gluingDiagram,
  },
};
