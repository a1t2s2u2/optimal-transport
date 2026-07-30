# 球面sourceのsensor多様体選択

`sensor_sphere_manifold.py` が現論文の再現scriptである。

~~~sh
uv run --python 3.12 sensor_sphere_manifold.py
~~~

学習器へ渡すのは $32\times32\times3$ のsensor画像と、第一channelから計算する校正済み距離だけである。
真のsource座標 $z\in S^2$ は診断評価にしか使わない。PEP 723 metadataによりNumPy、SciPy、
PyTorch、Matplotlibを一時環境へ導入する。

## 出力

- `sensor_source_to_image.png`: 球面上のsourceからsensor画像が生じる対応
- `sensor_examples.png`: networkへ与える入力画像
- `sensor_reconstructions.png`: 各潜在候補による同一画像の再構成
- `sensor_accuracy.png`: sensor数とnoiseに対する観測metricのRMSE
- `sensor_diagnostics.png`: 学習loss、真の距離と潜在距離、labelなし選択score
- `sensor_manifold_results.csv`: 全modelの評価値
- `sensor_manifold_history.csv`: 選択された各初期値の50 stepごとの学習履歴
- `sensor_distance_scatter.csv`: 真の距離と潜在距離の描画標本
- `sensor_resolution_study.csv`: sensor数を変えた観測metric精度
- `sensor_noise_study.csv`: noiseを変えた観測metric精度
- `sensor_manifold_table.tex`: 論文用の結果表

固定値は学習900枚、検証300枚、画像 $32\times32\times3$、AdamW 1,500 step、batch size 96、
MGW係数0.55、各model 2初期値、seed 20260730である。
