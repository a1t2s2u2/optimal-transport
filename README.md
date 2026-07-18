# Computational Optimal Transport — セミナー資料

最適輸送のセミナー発表資料（TeX / Web サイト）を管理するリポジトリ。

## セミナー

### `seminar/cuturi/` — Computational Optimal Transport

Peyré–Cuturi の教科書に沿い、最適輸送の計算的側面を扱う。

- 本編 4 章 + 付録 3 章
- 参考文献: [Computational Optimal Transport, G. Peyré & M. Cuturi (2019)](https://arxiv.org/abs/1803.00567)

### `seminar/wasserstein/` — Wasserstein 距離

Wasserstein 距離の定義・距離性の証明から W₂ における Gaussian 測度の話題までを目標とする。
複数の文献を横断的に参照し、現代的な記法で再構成する。

- 本編 4 章（導入 / Wₚ の定義と距離性 / Gaussian 集中 / 潜在空間への応用）+ 付録 1 章（距離空間と測度）
- 参考文献:
  - [A class of Wasserstein metrics for probability distributions, C. R. Givens & R. M. Shortt (1984)](https://doi.org/10.1307/mmj/1029003026)
  - [Optimal Transport: Old and New, C. Villani (2009)](https://doi.org/10.1007/978-3-540-71050-9)
  - [最適輸送理論とリッチ曲率, 桑江ほか (Encounter with Mathematics 第63回, 2015)](https://www.math.chuo-u.ac.jp/ENCwMATH/EwM63resume.pdf)

## ディレクトリ構成

```
seminar/
  cuturi/
    tex/          # TeX ソース（source of truth）
    site/         # Web サイト（tex から生成）
    reference/    # 参考文献 PDF
  wasserstein/
    tex/          # TeX ソース（source of truth）
    site/         # Web サイト（tex から生成）
    reference/    # 参考文献 PDF
```

## ビルド

```sh
# Cuturi
make cuturi-site        # Web サイト生成
make cuturi-pdf         # PDF 生成

# Wasserstein
make wasserstein-site   # Web サイト生成
make wasserstein-pdf    # PDF 生成
```
