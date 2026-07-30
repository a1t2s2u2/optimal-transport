# 観測画像から潜在多様体を選べるか

高次元観測を生成した潜在多様体を、有限個の候補から
Monge--Gromov--Wasserstein (MGW) 基準で選ぶ理論・実験論文。

## 一つの主張

encoder $E:X\to Z$ が定める距離歪み

$$
\mathcal D_2(E)^2
=\mathbb E\left[
  |d_X(X,X')-d_Z(E(X),E(X'))|^2
\right]
$$

は、観測空間と潜在空間の Gromov--Wasserstein 距離の上界になる。
独立な検証標本で有限個の潜在構造を比較すれば、選択誤差は
$O(\sqrt{\log m/n})$ で制御できる。

## 実験

未知の回転 $R\in SO(3)$ から、RGB 3軸markerを3台のcameraで観測した
$32\times32$画像を生成する。モデルに姿勢labelは与えない。

- 観測次元: 3,072
- 学習画像: 900枚
- 独立検証画像: 300枚
- 潜在候補: $\mathbb R^3$, $S^3$, $SO(3)$
- 深層モデル: 同一規模のconvolutional autoencoder

共通の検証scoreは $SO(3)$+MGW で最小になり、評価時だけ使った真の回転距離でも
最小stressを得る。

## 再現

実験のみ:

~~~sh
make paper-experiments
~~~

実験からPDFまで:

~~~sh
make paper-all
~~~

`uv` が PEP 723 metadata から Python 3.12 と依存関係を構築する。
PDF は `out/main.pdf` に生成される。
