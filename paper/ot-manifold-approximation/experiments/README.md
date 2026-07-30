# 球面から生成した応答画像の潜在構造選択

`sensor_sphere_manifold.py` が論文の再現scriptです。

~~~sh
uv run --python 3.12 sensor_sphere_manifold.py
~~~

モデルへ渡すのは $32\times32\times3$ の画像と、第一チャネルから計算する画像間距離です。
真の状態 $z\in S^2$ は診断評価にしか使いません。PEP 723 metadataによりNumPy、PyTorch、
Matplotlibを一時環境へ導入します。

## 出力

- `sensor_source_to_image.png`: 潜在状態から入力画像が生じる対応
- `sensor_examples.png`: モデルへ与える入力画像
- `sensor_reconstructions.png`: 各候補による同一画像の再構成
- `sensor_accuracy.png`: 画像解像度とノイズに対する距離RMSE
- `sensor_diagnostics.png`: 学習曲線、真の距離と潜在距離、候補選択の評価値
- `sensor_manifold_results.csv`: 全候補の評価値
- `sensor_manifold_history.csv`: 50 stepごとの学習履歴
- `sensor_distance_scatter.csv`: 真の距離と潜在距離の描画標本
- `sensor_resolution_study.csv`: 画像解像度を変えた距離精度
- `sensor_noise_study.csv`: ノイズを変えた距離精度
- `sensor_manifold_table.tex`: 論文用の結果表

固定値は学習900枚、検証300枚、画像 $32\times32\times3$、AdamW 1,500 step、batch size 96、
距離損失係数0.55、各モデル2初期値、seed 20260730です。
