# 多視点画像からの潜在多様体選択

`rotation_image_manifold.py` が現論文の再現scriptである。

~~~sh
uv run --python 3.12 rotation_image_manifold.py
~~~

PEP 723 metadata により NumPy、SciPy、PyTorch、Matplotlib を一時環境へ導入する。
学習に使うのは画像とpixel近傍graphだけで、真の回転quaternionは診断評価にしか使わない。

## 出力

- `rotation_examples.png`: labelを隠した入力画像
- `rotation_reconstructions.png`: 各潜在構造の再構成
- `rotation_diagnostics.png`: loss、距離対応、stress比較
- `rotation_manifold_results.csv`: 全評価値
- `rotation_manifold_history.csv`: 50 stepごとの学習履歴
- `rotation_distance_scatter.csv`: 真の回転距離と学習距離の描画標本
- `rotation_manifold_table.tex`: 論文用の結果表

固定値は学習900枚、検証300枚、画像$32\times32\times3$、AdamW 1500 step、
batch size 96、MGW係数0.35、seed 20260730である。

同じdirectoryに残る球面・局所decoder実験は旧版論文の再現物であり、現論文の主張や
`make paper-all`には使わない。
