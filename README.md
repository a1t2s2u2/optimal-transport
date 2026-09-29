# Computational Optimal Transport — セミナー資料

最適輸送のセミナー発表資料（TeX / Web サイト）を管理するリポジトリ。

## セミナー

### `seminar/cuturi/` — Computational Optimal Transport

Peyré–Cuturi の教科書に沿い、最適輸送の計算的側面を扱う。

- 本編 4 章 + 付録 4 章
- 参考文献: [Computational Optimal Transport, G. Peyré & M. Cuturi (2019)](https://arxiv.org/abs/1803.00567)

### `seminar/wasserstein/` — Wasserstein 距離

Wasserstein 距離を定義し、最適 coupling の存在を経て距離性を証明する。

- 本編 2 章（導入 / Wₚ の距離性）+ 付録 1 章（距離空間と測度）
- 参考文献:
  - [Optimal Transport: Old and New, C. Villani (2009)](https://doi.org/10.1007/978-3-540-71050-9)
  - [最適輸送理論とリッチ曲率, 桑江ほか (Encounter with Mathematics 第63回, 2015)](https://www.math.chuo-u.ac.jp/ENCwMATH/EwM63resume.pdf)

### `seminar/quantum-ot/` — 量子最適輸送

有限次元量子系の状態・coupling・チャネルを導入し、多体系の量子 Wasserstein 距離
`W₁` の距離性と、対角状態上での Hamming–Wasserstein 距離への還元を示す。

- 本編 5 章（量子状態 / coupling / チャネル / 量子 `W₁` / 古典系への還元）
- 参考文献:
  - [Quantum Optimal Transport: Quantum Channels and Qubits, De Palma–Trevisan (2023)](https://arxiv.org/abs/2307.16268)
  - [The Quantum Wasserstein Distance of Order 1, De Palma et al. (2021)](https://arxiv.org/abs/2009.04469)

## ディレクトリ構成

```
tools/
  site/               # tex → Web サイトの変換エンジン（全セミナー共通）
seminar/
  cuturi/
    tex/              # TeX ソース（source of truth）
    site.config.mjs   # このセミナーのサイト設定
    site/             # Web サイト（tex から生成・git 管理外）
    reference/        # 原典の対訳と正誤表
  wasserstein/
    tex/
    site.config.mjs
    site/
    reference/
  quantum-ot/
    tex/
    site.config.mjs
    site/
```

サイト生成のエンジンは `tools/site/` に 1 つだけ置き、セミナーごとの章立て・タイトル・
固有の図は `site.config.mjs`、数式マクロは各 `tex/preamble.tex` に置く。
詳細は [`tools/site/README.md`](tools/site/README.md)。

## ビルド

```sh
make sites              # すべてのセミナーの Web サイトを生成
make cuturi-site        # 計算最適輸送の Web サイト
make wasserstein-site   # Wasserstein 距離の Web サイト
make quantum-ot-site    # 量子最適輸送の Web サイト
```

`main` へ push すると `.github/workflows/pages.yml` が tex からサイトを生成し
GitHub Pages へデプロイする。生成物をコミットする必要はない。
