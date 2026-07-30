# 画像からロボットの関節空間を見抜けるか

2関節robot armの画像だけから、背後のconfiguration space
$S^1\times S^1=\mathbb T^2$を有限候補中から選ぶ理論・実験論文。

## 一つの主張

encoder $E:X\to Z$ が定める距離歪み

$$
\mathcal D_2(E)^2
=\mathbb E\left[
  |d_X(X,X')-d_Z(E(X),E(X'))|^2
\right]
$$

はGromov--Wasserstein距離の上界になる。観測metricの誤差を
$\varepsilon_{\mathrm{obs}}$とすれば、真の多様体距離に対する潜在距離の誤差は
$\mathcal D_2(E)+\varepsilon_{\mathrm{obs}}$以下である。独立検証標本による有限候補の
選択誤差は $O(\sqrt{\log m/n})$ で制御できる。

## 実験

- 観測: 色分けした2関節armの $32\times32$ RGB画像
- 学習画像: 900枚
- 独立検証画像: 300枚
- 潜在候補: $\mathbb R^2$, $S^2$, $\mathbb T^2$
- 深層モデル: 同一規模のconvolutional autoencoder
- 学習には画像とpixel 4近傍graphだけを使用
- 関節角は診断評価にだけ使用

共通のlabelなし検証scoreは $\mathbb T^2$+MGW で最小になる。ただし次点との差は小さく、
真の関節metricでは $S^2$ が勝つため、観測metric精度が未解決のボトルネックである。

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
