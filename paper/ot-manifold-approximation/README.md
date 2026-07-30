# 高次元sensor画像から潜在多様体をどこまで復元できるか

低次元source $z\in S^2$ が $32\times32\times3=3{,}072$ 次元のsensor画像を生成する。
source座標を見ないCNNが、候補 $\mathbb R^2,\mathbb T^2,S^2$ のうち正しい潜在構造と
その距離を選べる条件を、有限sensor誤差とMonge--Gromov--Wasserstein (MGW) stressで記述する
理論・実験論文である。

## 一つの主張

校正channel

$$
r_z(a)=\frac12+\alpha\langle a,z\rangle
$$

の無限sensor $L^2$ 距離は、$S^2$ の正規化chordal距離に厳密に一致する。$D$個の有限sensorでは、
二乗距離の一様誤差がsensor方向の二次moment行列 $Q_D$ により

$$
\sup_{z,z'}\left|\widehat d_D(z,z')^2-d_\star(z,z')^2\right|
\le 3\left\|Q_D-I_3/3\right\|_{\mathrm{op}}
$$

と評価できる。緯度経度midpoint配置なら右辺は $O(D^{-1})$ である。この観測誤差に、
MGW stressと独立検証標本の誤差を加えることで、正しい候補多様体を選べるgap条件を得る。
この結果はoptimizerや誤差逆伝播法を仮定しない。

## 実験

- 潜在source: $S^2$ 上の一点
- 観測: 1,024 sensorによる $32\times32\times3$ response画像
- 学習画像900枚、独立検証画像300枚
- 候補: $\mathbb R^2$, $\mathbb T^2$, $S^2$
- 深層model: 同一規模、約41万parameterのconvolutional autoencoder
- 学習・選択にsource座標を使わず、校正channelから得る観測距離を使用

labelなし検証scoreは $S^2$+MGWで $0.00079$、$\mathbb T^2$+MGWで $0.00620$、
$\mathbb R^2$+MGWで $0.06190$となる。選択modelの真の距離stressは $0.00112$、
10近傍recallは $0.994$ である。sensor数を16から4,096へ変えた実験ではmetric RMSEが
ほぼ $D^{-1}$ で減少する。

## 再現

実験のみ:

~~~sh
make paper-experiments
~~~

実験からPDFまで:

~~~sh
make paper-all
~~~

`uv` がPEP 723 metadataからPython 3.12と依存関係を構築する。
PDFは `out/main.pdf` に生成される。
