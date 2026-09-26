# Computational Optimal Transport — セミナー資料

最適輸送のセミナー発表資料（TeX / Web サイト）を管理するリポジトリ。

理論論文は `paper/ot-manifold-approximation/` に置き、数値実験は含めない。

## セミナー

### `seminar/cuturi/` — Computational Optimal Transport

Peyré–Cuturi の教科書に沿い、最適輸送の計算的側面を扱う。

- 本編 4 章 + 付録 3 章
- 参考文献: [Computational Optimal Transport, G. Peyré & M. Cuturi (2019)](https://arxiv.org/abs/1803.00567)

### `seminar/wasserstein/` — Wasserstein 距離

Wasserstein 距離を定義し、最適 coupling の存在を経て距離性を証明する。

- 本編 2 章（導入 / Wₚ の距離性）+ 付録 1 章（距離空間と測度）
- 参考文献:
  - [Optimal Transport: Old and New, C. Villani (2009)](https://doi.org/10.1007/978-3-540-71050-9)
  - [最適輸送理論とリッチ曲率, 桑江ほか (Encounter with Mathematics 第63回, 2015)](https://www.math.chuo-u.ac.jp/ENCwMATH/EwM63resume.pdf)

## ディレクトリ構成

```
tools/
  site/               # tex → Web サイトの変換エンジン（全セミナー共通）
paper/
  ot-manifold-approximation/ # Wasserstein 潜在幾何の理論論文（英語・日本語）
seminar/
  cuturi/
    tex/              # TeX ソース（source of truth）
    site.config.mjs   # このセミナーのサイト設定
    site/             # Web サイト（tex から生成・git 管理外）
    reference/        # 参考文献 PDF
  wasserstein/
    tex/
    site.config.mjs
    site/
    reference/
```

サイト生成のエンジンは `tools/site/` に 1 つだけ置き、セミナーごとの違い
（章立て・タイトル・用語集・数式マクロ）は `site.config.mjs` に閉じ込める。
詳細は [`tools/site/README.md`](tools/site/README.md)。

## ビルド

```sh
make sites              # すべてのセミナーの Web サイトを生成
make cuturi-site        # 計算最適輸送の Web サイト
make cuturi-pdf         # 計算最適輸送の PDF
make wasserstein-site   # Wasserstein 距離の Web サイト（PDF は生成しない）
make paper-all          # 理論論文の英語版・日本語版 PDF
```

`main` へ push すると `.github/workflows/pages.yml` が tex からサイトを生成し
GitHub Pages へデプロイする。生成物をコミットする必要はない。
