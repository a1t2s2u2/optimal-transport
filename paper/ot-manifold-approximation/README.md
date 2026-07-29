# ニューラル局所チャートの Wasserstein 表現限界

多様体データを生成するために、局所 decoder が何個必要かを扱う理論論文。
categorical chart、chart 内の neural flow、affine tangent decoder からなる
$K$-chart tangent-flow generator を主対象とする。

## モデル

$$
J\sim\operatorname{Categorical}(\pi),\qquad
Z_0\sim\mathcal N(0,I_q),
$$

$$
Z_1=\Phi_{v_J}(Z_0),\qquad
X=x_J+U_JZ_1,qquad
\operatorname{ran}U_J=T_{x_J}M.
$$

- $J$: chart gating
- $v_J$: chart 内の潜在分布を生成する Flow Matching / neural ODE
- $x_J+U_Jz$: データ空間へ写す affine tangent decoder

このモデルの最良 $W_p$ 生成誤差を $\mathfrak N_{K,p}$ と定義する。

## 主結果

局所 flow が $W_p$ の意味で普遍的なら、ニューラル生成モデルの近似幅は接アトラス幅と
厳密に一致する。

$$
\mathfrak N_{K,p}(\mu;M)
=\mathfrak T_{K,p}(\mu;M).
$$

したがって、生成誤差は

$$
W_p(\mu,\widehat\nu)
\le
\text{decoder の幾何誤差}
+\text{局所 flow の分布誤差}
$$

と分離できる。reach、Ahlfors 正則性、quadratic tangent separation のもとでは

$$
cK^{-2/q}
\le \mathfrak N_{K,p}(\mu;M)
\le CK^{-2/q}.
$$

よって誤差 $\varepsilon$ に必要な affine decoder 数は
$K\asymp\varepsilon^{-q/2}$。Stiefel、Grassmann、積 torus で条件を検証し、
$S^1$ では厳密値まで計算する。

これは任意の深層ネットに対する下界ではなく、真の接空間を使う
tangent-constrained generator の oracle 表現限界である。

## 数値検証

円周、球面、積 torus、Stiefel、Grassmann の5例について、式の幾何誤差だけを直接
積分し、有限 $K$ での log--log 傾きを確認する。さらに円周の $K=8$ chart内では、
小型 tanh MLP の速度場を conditional Flow Matching で実際に学習し、局所flow誤差と
decoder幾何floorを分離する。

```sh
make paper-experiments
```

詳細は `experiments/README.md` を参照。

## ビルド

```sh
make paper
```

実験を再実行してから PDF を生成する場合は `make paper-all`。
出力は `paper/ot-manifold-approximation/out/main.pdf`。
