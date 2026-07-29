# 曲率制御付き Wasserstein 潜在拡散

VAE の潜在コードではなく、decoder が表す条件付き分布を $W_2$ に関して
等方的に拡散する latent diffusion model の理論草稿。

## 中心仮説

標準的な latent diffusion の Euclid noise は、decoder 分布上では非一様である。
対角 Gaussian decoder

$$
K_z=\mathcal N(m(z),\operatorname{diag}(\sigma(z)^2))
$$

に対して

$$
G(z)=J_m(z)^\top J_m(z)+J_\sigma(z)^\top J_\sigma(z)
$$

とおくと、標準 noise の decoded $W_2^2$ 変化率は
$\beta\operatorname{tr}G(z)$ になる。共分散を $\beta G(z)^{-1}$ にすると、
変化率は潜在点によらず $\beta d$ となる。

## 主な主張

- $G^{-1}$ は decoder manifold の接空間上で decoded noise を等方化する唯一の共分散
- VAE prior を厳密な不変分布にする座標不変な forward generator
- forward law は $(\mathcal Z,G)$ 上の KL の Wasserstein 勾配流
- 第二基本形式の trace は decoded process の平均曲率 drift
- 曲率証明書 $\Lambda$ は geodesic step と decoded $W_2$ の差を三次で抑える
- natural relative score は matrix-free な Riemannian score matching で学習可能

## 新規性の境界

$W_2$ 引き戻し計量そのものは *Optimal Latent Transport* (Roy & Hauberg, 2022)
に先行研究がある。一般の Riemannian score model も既知である。
本稿の対象は、decoder-$W_2$ の等方性から noise covariance を一意に導き、
prior drift・平均曲率 drift・曲率適応 step まで同じ幾何から設計することである。

## 状態

- 理論草稿
- 実験は未実施
- 最初の検証対象は synthetic decoder と低次元 MNIST VAE
- 大規模画像 LDM には metric inverse の近似または蒸留が必要

## ビルド

```sh
make paper
```

出力は `out/main.pdf`。uplatex + dvipdfmx を使用する。
