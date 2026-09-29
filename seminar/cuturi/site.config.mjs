// 計算最適輸送セミナーのサイト設定。
// 変換エンジン本体は tools/site/。ここにはこのセミナー固有の情報だけを置く。
// 数式マクロは tex/preamble.tex から自動抽出されるので、ここに書き写す必要はない。

import { transportCostDiagram, transportOptimalDiagram } from "./site-demos.mjs";

export default {
  // --- 表示 ---
  title: "計算最適輸送",
  logo: "OT",
  siteName: "計算最適輸送セミナー",
  landingTitle: "計算最適輸送",
  landingSubtitle: "Computational Optimal Transport — G. Peyré &amp; M. Cuturi",
  landingFooter:
    "Based on <em>Computational Optimal Transport</em> by G. Peyr&eacute; &amp; M. Cuturi",
  appendixHeading: "付録：前提知識",
  appendixSubheading: "発表では省略した数学的前提を網羅した完全版．本編から参照される．",

  // --- 章立て（tex は tex/ からの相対パス、md は site/content/ に出る）---
  chapters: [
    {
      tex: "main/01_assignment.tex",
      md: "01-assignment.md",
      id: "assignment",
      group: "main",
      nav: "最適割当",
      eyebrow: "1. Assignment",
      title: "最適割当問題",
    },
    {
      tex: "main/02_monge.tex",
      md: "02-monge.md",
      id: "monge",
      group: "main",
      nav: "Monge",
      eyebrow: "2. Monge",
      title: "Monge 問題",
    },
    {
      tex: "main/03_kantorovich.tex",
      md: "03-kantorovich.md",
      id: "kantorovich",
      group: "main",
      nav: "Kantorovich",
      eyebrow: "3. Kantorovich",
      title: "Kantorovich 問題",
    },
    {
      tex: "main/04_entropic.tex",
      md: "04-entropic.md",
      id: "entropic",
      group: "main",
      nav: "エントロピー",
      eyebrow: "4. Entropic Regularization",
      title: "エントロピー正則化",
    },
    {
      tex: "foundations/00_set_topology.tex",
      md: "A0-set-topology.md",
      id: "found-set-topology",
      group: "appendix",
      nav: "集合と位相",
      eyebrow: "付録 A. Set & Topology",
      title: "集合と位相",
    },
    {
      tex: "foundations/01_metric_compact.tex",
      md: "A1-metric.md",
      id: "found-metric",
      group: "appendix",
      nav: "距離・コンパクト",
      eyebrow: "付録 B. Metric & Compactness",
      title: "距離空間・連続・コンパクト性",
    },
    {
      tex: "foundations/02_measure.tex",
      md: "A2-measure.md",
      id: "found-measure",
      group: "appendix",
      nav: "測度論",
      eyebrow: "付録 C. Measure Theory",
      title: "測度論",
    },
    {
      tex: "foundations/03_convex_linalg.tex",
      md: "A3-convex.md",
      id: "found-convex",
      group: "appendix",
      nav: "凸・線形代数",
      eyebrow: "付録 D. Convexity & Linear Algebra",
      title: "凸性と線形代数",
    },
  ],

  // MathJax は stmaryrd を持たないため、\range だけ等価な綴りに差し替える。
  macroOverrides: {
    range: ["{\\lbrack\\!\\lbrack}#1{\\rbrack\\!\\rbrack}", 1],
  },

  // --- tex 側の \demohint{名前} から差し込む図 ---
  demos: {
    "transport-cost": transportCostDiagram,
    "transport-optimal": transportOptimalDiagram,
  },
};
