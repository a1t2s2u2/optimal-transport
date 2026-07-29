# 接アトラスの Wasserstein 近似幅

低次元多様体上の確率測度を、有限個の接平面からなるアトラスで
どこまで近似できるかを扱う理論論文。一般の positive-reach 多様体を主対象とし、
Stiefel、Grassmann、積 torus を明示例として扱う。

## 中心問題

埋め込み多様体 $M\subset\mathbb R^D$ と $\mu\in\mathcal P_p(M)$ に対し、
$K$ 個の点 $x_j\in M$ の affine tangent plane

$$
A(x_1,\ldots,x_K)=\bigcup_{j=1}^K(x_j+T_{x_j}M)
$$

で支えられる測度までの最良 Wasserstein 誤差

$$
\mathfrak T_{K,p}(\mu;M)
=\inf_{x_1,\ldots,x_K}
  \inf_{\operatorname{supp}\nu\subset A(x_1,\ldots,x_K)}
  W_p(\mu,\nu)
$$

を tangent-atlas Wasserstein width と定義する。

## 主結果

- 閉集合への最良 Wasserstein 近似は距離関数の $L^p$ ノルムに一致する
- reach が正の一般多様体では、幅は covering radius の二乗以下
- Ahlfors 正則性と quadratic tangent separation のもとで $K^{-2/q}$ の
  matching lower bound
- $M=\operatorname{St}(D,r)$、$q=Dr-r(r+1)/2$、$\sigma=$ Haar 分布なら

  $$
  c_{D,r,p}K^{-2/q}
  \le \mathfrak T_{K,p}(\sigma;M)
  \le C_{D,r,p}K^{-2/q}
  $$

  が成り立つ
- $\operatorname{St}(2,1)=S^1$、$p=2$ では等間隔配置が厳密最適で、定数まで計算できる
- Grassmann 多様体と積 torus でも同じ sharp rate
- 幅は入力測度について $W_p$-Lipschitz であり、経験測度への置換誤差を分離できる

本稿は最適化法、ニューラルネット、誤差逆伝播法を仮定しない。

## 数値検証

円周、球面、積 torus、Stiefel、Grassmann の5例について、有限 $K$ での
log--log 傾きが理論値 $-2/q$ に近づくかを直接積分で確認する。外部 Python
パッケージや勾配法は使わない。

```sh
make paper-experiments
```

実験の設計と生成物は `experiments/README.md` を参照。

## ビルド

```sh
make paper
```

実験を再実行してから PDF を生成する場合は `make paper-all`。

出力は `paper/ot-manifold-approximation/out/main.pdf`。
