# 2関節arm画像からの潜在多様体選択

`articulated_arm_manifold.py` が現論文の再現scriptである。

~~~sh
uv run --python 3.12 articulated_arm_manifold.py
~~~

学習に使うのは画像とpixel近傍graphだけで、真の二関節角は診断評価にしか使わない。
PEP 723 metadataによりNumPy、SciPy、PyTorch、Matplotlibを一時環境へ導入する。

## 出力

- `arm_torus_explainer.png`: 二つの関節周期とtorus上の二loopの対応
- `arm_examples.png`: 角度labelを隠した入力画像
- `arm_reconstructions.png`: 各潜在構造の再構成
- `arm_diagnostics.png`: loss、距離対応、検証score
- `arm_manifold_results.csv`: 全評価値
- `arm_manifold_history.csv`: 選ばれた初期値の50 stepごとの学習履歴
- `arm_distance_scatter.csv`: 真のtorus距離と学習距離の描画標本
- `arm_manifold_table.tex`: 論文用の結果表

固定値は学習900枚、検証300枚、画像$32\times32\times3$、pixel 4近傍graph、
AdamW 1500 step、batch size 96、MGW係数0.35、各モデル2初期値、seed 20260730である。
