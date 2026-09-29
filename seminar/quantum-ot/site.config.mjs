// 量子最適輸送セミナーのサイト設定。
// 数式マクロは tex/preamble.tex から自動抽出される。

export default {
  title: "量子最適輸送",
  logo: "QOT",
  siteName: "量子最適輸送セミナー",
  landingTitle: "量子最適輸送",
  landingSubtitle: "量子状態とチャネルから Hamming–Wasserstein W₁ の距離性まで",
  landingFooter:
    '参考文献: <a href="https://arxiv.org/abs/2307.16268">De Palma–Trevisan (2023)</a>, <a href="https://arxiv.org/abs/2009.04469">De Palma et al. (2021)</a>',

  chapters: [
    {
      tex: "main/00_quantum_states.tex",
      md: "00-quantum-states.md",
      id: "quantum-states",
      nav: "量子状態",
      eyebrow: "0. Quantum States",
      title: "有限次元の量子状態",
    },
    {
      tex: "main/01_quantum_couplings.tex",
      md: "01-quantum-couplings.md",
      id: "quantum-couplings",
      nav: "量子 coupling",
      eyebrow: "1. Quantum Couplings",
      title: "量子 coupling",
    },
    {
      tex: "main/02_quantum_channels.tex",
      md: "02-quantum-channels.md",
      id: "quantum-channels",
      nav: "量子チャネル",
      eyebrow: "2. Quantum Channels",
      title: "輸送計画としての量子チャネル",
    },
    {
      tex: "main/03_quantum_w1.tex",
      md: "03-quantum-w1.md",
      id: "quantum-w1",
      nav: "量子 W₁",
      eyebrow: "3. Quantum W₁",
      title: "多体系の量子 Wasserstein 距離 W₁",
    },
    {
      tex: "main/04_classical_reduction.tex",
      md: "04-classical-reduction.md",
      id: "classical-reduction",
      nav: "古典系への還元",
      eyebrow: "4. Classical Reduction",
      title: "Hamming–Wasserstein 距離への還元",
    },
  ],
};
