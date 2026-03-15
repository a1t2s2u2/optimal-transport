# 最適輸送理論の解説書

## 概要

最適輸送理論を数学的基盤から体系的に解説するドキュメントを作成する。
最終的にファイバー束上の最適輸送距離まで到達し、機械学習への応用を議論する。

## 関連リポジトリ

- [manifold-optimization](https://github.com/a1t2s2u2/manifold-optimization) — Stiefel多様体上の最適化の解説書。最適輸送との接点（WGAN のLipschitz制約）を第8節で簡潔に触れている。

## ドキュメント構成（案）

```
note/
  main.tex              ← プリアンブル + \input で各セクションを読み込む
  sections/
    sec1_measure.tex     ← 測度論の基礎（確率測度、押し出し測度、弱収束）
    sec2_monge.tex       ← Monge 問題（輸送写像、存在の困難）
    sec3_kantorovich.tex ← Kantorovich 緩和（輸送計画、線形計画、双対性）
    sec4_wasserstein.tex ← Wasserstein 距離（p-Wasserstein、位相的性質、KR双対定理）
    sec5_computation.tex ← 計算手法（エントロピー正則化、Sinkhorn アルゴリズム）
    sec6_fiber.tex       ← ファイバー束上の最適輸送（距離ファイバー束、構造化コスト）
    sec7_ml.tex          ← 機械学習への応用（WGAN、ドメイン適応、点群比較）
    sec8_conclusion.tex  ← 結論
```

## 方針

- manifold-optimization の解説書と同じスタイル（uplatex, tcolorbox定理環境）を使う
- 数学的厳密性を重視：定義→定理→証明の形式で記述
- 概念ごとに図を用意して直観を補う
- 各節冒頭に「なぜこの概念が必要か」の動機を書く
- コミットメッセージに Co-Authored-By は不要

## TeX コンパイル

```bash
cd note && uplatex -output-directory=out main.tex
```

## .gitignore

```
note/out/
*.pdf
```

## 参考資料

- Villani, "Optimal Transport: Old and New" (2008)
- Peyré & Cuturi, "Computational Optimal Transport" (2019)
- YouTube: https://www.youtube.com/watch?v=IZ10Ih2e1cM （p-Monge-Kantorovich の解説）
