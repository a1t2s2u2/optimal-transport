# 数値実験

主定理 $\mathfrak T_{K,2}\asymp K^{-2/q}$ の有限 $K$ での挙動を確認する再現用コード。
接平面の合併までの $W_2$ 距離を直接積分する。ニューラルネットの訓練実験ではなく、
理論式の数値検算である。

加えて `neural_flow_circle.py` は、$S^1$ の $K=8$ 局所 decoder 内で小型 tanh
MLP の速度場を conditional Flow Matching により実際に学習する。Adam 更新と neural
ODE の Heun 積分も標準ライブラリだけで実装している。これは、潜在分布を学習しても
decoder の幾何誤差が残ることを確認する補助実験である。

主たる学習実験は `sphere_multimodal_flow.py` である。$S^2$ 上の6個の
接平面 decoder に各3 modeを配置し、全体で18峰性の分布を作る。共有MLPはchartの
one-hot表現を条件として二次元局所速度場を学習する。生成品質は48方向の局所
Sliced-$W_2$ と、球面上の目標・生成点群で評価する。

```sh
make paper-experiments
```

生成物は次の通り。

- `results.csv`: 各 $K$ での $W_2$ 上界
- `slopes.csv`: 最後の4点に対する log--log 回帰
- `results_table.tex`: 論文に取り込む表
- `sphere_rate.csv`: $S^2$ のdecoder数と誤差、および $K^{-1}$ 基準線
- `sphere_generated_K*.csv`: $K=4,16,64$ の球面decoder生成点
- `sphere_flow_results.csv`: 18峰性球面分布のloss、局所SW$_2$、半径誤差
- `sphere_flow_history_width*.csv`: 球面Flowの100 stepごとの学習履歴
- `sphere_flow_outputs.csv`: 球面上の目標点と各幅の生成点
- `neural_flow_results.csv`: MLP 幅ごとの潜在 $W_2$ と生成誤差上界
- `neural_flow_table.tex`: neural flow 実験の論文用表
- `neural_flow_history_width*.csv`: 100 step ごとの学習 loss EMA と固定検証 loss
- `neural_flow_quantiles.csv`: 目標分布と各 MLP が生成した終端分布の分位点

単純な $S^1$ の履歴と分位点もCSVには残すが、論文の主図には使わない。
loss は conditional regression の既約分散を含むため、生成品質は
`neural_flow_results.csv` の終端 $W_2$ と併せて読む。

中心は、$S^1$ では厳密最適な等間隔配置、$S^2$ では Fibonacci 配置、
$\mathbb T^2$ では直積格子を使う。$\operatorname{St}(3,2)$ と
$\operatorname{Gr}(4,2)$ では、固定 seed で生成した Haar 標本候補に対する
farthest-point sampling (FPS) を使う。後二者を含め、得られる値は最適幅そのものでは
なく、具体的な配置が達成する上界である。

再現性のため seed、標本数、$K$ は `run.py` に固定している。標準ライブラリだけで
実行できる。neural flow 側の seed、学習step、batch size、ODE stepも
`sphere_multimodal_flow.py` と `neural_flow_circle.py` に固定している。
