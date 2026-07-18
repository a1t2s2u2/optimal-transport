# Cuturi-Peyré "Computational Optimal Transport" 既知の誤り

本リポジトリ `cuturi/original/` と `cuturi/translation/` は原典に忠実に保持しており，以下の数学的誤りも修正していない．`seminar/` 側では正しい記述を採用している．

## 1. §3.1 LP 標準形における Kronecker 積の不整合

**該当箇所**：
- `cuturi/original/chapters/algo-basics.tex` L19-25
- `cuturi/translation/03_algorithmic_foundations.tex` L34-38

**原典の記述**：

行列 $\mathbf{P} \in \R^{n \times m}$ を **列方向**ベクトル化し（$\mathbf{p}$ の第 $i + n(j-1)$ 成分が $P_{i,j}$），制約行列を

$$
\mathbf{A} = \begin{bmatrix}
  \ones_n^\top \otimes \Identity_m \\
  \Identity_n \otimes \ones_m^\top
\end{bmatrix}
\in \R^{(n+m) \times nm}
$$

と定めるとしている．

**問題点**：

具体的に $n=3, m=2$ で検算する．列方向 vec で
$\mathbf{p} = (P_{1,1}, P_{2,1}, P_{3,1}, P_{1,2}, P_{2,2}, P_{3,2})^\top$．

上段ブロック $\ones_3^\top \otimes \Identity_2 = [I_2 \mid I_2 \mid I_2]$（$2 \times 6$ 行列）の第 1 行は
$(1, 0, 1, 0, 1, 0)$ であり，
$\mathbf{p}$ に作用させると
$p_1 + p_3 + p_5 = P_{1,1} + P_{3,1} + P_{2,2}$
となる．これは **行和でも列和でもない混合**であり，
意図された $\mathbf{P}\ones_m = \mathbf{a}$ または $\mathbf{P}^\top \ones_n = \mathbf{b}$ と
整合しない．

**正しい記述**：

列方向 vec のもとでは
$$
\mathbf{A} = \begin{bmatrix}
  \ones_m^\top \otimes \Identity_n \\
  \Identity_m \otimes \ones_n^\top
\end{bmatrix}
$$
とすれば，上段が行和制約 $\mathbf{P}\ones_m = \mathbf{a}$（$n$ 行）に，
下段が列和制約 $\mathbf{P}^\top\ones_n = \mathbf{b}$（$m$ 行）に対応する．

**確認（$n=3, m=2$）**：
- 上段 $\ones_2^\top \otimes \Identity_3 = [I_3 \mid I_3]$．第 1 行 $(1, 0, 0, 1, 0, 0)$ で $p_1 + p_4 = P_{1,1} + P_{1,2} = (\mathbf{P}\ones_m)_1$ ✓
- 下段 $\Identity_2 \otimes \ones_3^\top$．第 1 行 $(1, 1, 1, 0, 0, 0)$ で $p_1 + p_2 + p_3 = P_{1,1} + P_{2,1} + P_{3,1} = (\mathbf{P}^\top\ones_n)_1$ ✓

**注**：階数の議論「最初の $n$ 行と残りの $m$ 行の和は共に $\ones_{nm}^\top$」は，正しい formula のもとでも成立する（Cuturi 原典の証明文は誤った formula を前提とすると次元が合わないが，正しい formula なら自然に整合する）．

**seminar 側の対応**：`seminar/ch03_algorithmic_foundations.tex` §3.1 で正しい formula に修正済み．

---

## 検出経緯

PR #6（`refine/ch01-topology-section`）に対する Copilot レビュー（2026-04-28）で指摘された．seminar の §3.1 が原典 formula を忠実に翻訳していたため，原典側の誤りも継承していた．
