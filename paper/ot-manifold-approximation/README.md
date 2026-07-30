# 局所線形 decoder は何個必要か

多様体上の分布を、$K$ 個の局所線形 decoder でどこまで生成できるかを扱う理論論文。

## モデル

$$
J\sim\operatorname{Categorical}(\pi),\qquad
Z\mid J=j\sim\lambda_j,\qquad
X=x_J+U_JZ,
$$

$$
\operatorname{ran}U_j=T_{x_j}M.
$$

各枝の潜在分布 $\lambda_j$ は任意とし、decoder 数 $K$ だけを表現予算として測る。
このモデルの最良 $W_p$ 生成誤差を $\mathfrak T_{K,p}$ とする。

## 一つの主結果

positive reach、Ahlfors 正則性、quadratic tangent separation のもとで

$$
cK^{-2/q}
\le \mathfrak T_{K,p}(\mu;M)
\le CK^{-2/q}.
$$

したがって、誤差 $\varepsilon$ に必要な局所 decoder 数は

$$
K\asymp\varepsilon^{-q/2}.
$$

Stiefel、Grassmann、積 torus で仮定を検証し、$S^1$ では最適値を厳密に計算する。
これは任意の深層ネットに対する下界ではなく、真の接空間を使う局所線形 decoder の
oracle 表現限界である。

multi-branch VAE は枝 $J$ と潜在変数 $Z$ を学習する実装、Flow Matching は各 $\lambda_j$ を
学習する実装として解釈できる。ただし両者は主定理の仮定ではない。

## 数値検証

円周、球面、積 torus、Stiefel、Grassmann の5例で、有限 $K$ の log--log 傾きを
確認する。円周の小型 Flow Matching は、潜在分布を学習しても decoder の幾何誤差が
残ることを見る補助実験である。学習loss、終端分布の分位点、潜在 $W_2$ を保存し、
論文中にloss曲線と生成出力曲線を掲載する。

~~~sh
make paper-experiments
~~~

詳細は experiments/README.md を参照。

## ビルド

~~~sh
make paper
~~~

実験を再実行してから PDF を生成する場合は make paper-all。
出力は paper/ot-manifold-approximation/out/main.pdf。
