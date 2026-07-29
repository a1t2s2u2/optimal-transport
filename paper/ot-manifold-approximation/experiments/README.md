# 数値実験

式 (2.10) に現れる decoder 幾何誤差の、有限 $K$ での挙動を確認する再現用コード。
局所 neural flow が理想的に潜在分布を表現した後にも残る、接平面の合併までの
$W_2$ 距離を直接積分する。ニューラルネットの訓練実験ではなく、理論式の数値検算である。

```sh
make paper-experiments
```

生成物は次の通り。

- `results.csv`: 各 $K$ での $W_2$ 上界
- `slopes.csv`: 最後の4点に対する log--log 回帰
- `results_table.tex`: 論文に取り込む表

中心は、$S^1$ では厳密最適な等間隔配置、$S^2$ では Fibonacci 配置、
$\mathbb T^2$ では直積格子を使う。$\operatorname{St}(3,2)$ と
$\operatorname{Gr}(4,2)$ では、固定 seed で生成した Haar 標本候補に対する
farthest-point sampling (FPS) を使う。後二者を含め、得られる値は最適幅そのものでは
なく、具体的な配置が達成する上界である。

再現性のため seed、標本数、$K$ は `run.py` に固定している。標準ライブラリだけで
実行できる。
