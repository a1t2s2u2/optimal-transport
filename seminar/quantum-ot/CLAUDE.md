## seminar/quantum-ot

`seminar/wasserstein` の自然な続編として，有限次元の量子最適輸送を扱う。
主目標は，多体系の量子 Wasserstein 距離 $W_1$ を定義し，距離性と
古典 Hamming--Wasserstein 距離への還元を示すことである。

## 参考文献

- G. De Palma and D. Trevisan, *Quantum Optimal Transport: Quantum Channels and Qubits*, arXiv:2307.16268 — 全体構成の主たる文献
- G. De Palma, M. Marvian, D. Trevisan, and S. Lloyd, *The Quantum Wasserstein Distance of Order 1*, arXiv:2009.04469 — 量子 $W_1$ の原論文

## 記法

- 有限次元 Hilbert 空間だけを扱う
- $mathcal{L}(mathcal{H})$ は線形作用素，$mathcal{O}(mathcal{H})$ は Hermite 作用素，$mathcal{S}(mathcal{H})$ は量子状態
- 量子状態には $ho,\sigma,\tau$，coupling には $pi$，量子チャネルには $Phi$ を使う
- 部分跡は $operatorname{Tr}_{\mathcal{K}}$，跡ノルムは $|\cdot\|_1$ と書く
- $n$-qubit 系は $mathcal{H}_n=(\mathbb{C}^2)^{\otimes n}$，跡が $0$ の Hermite 作用素全体は $mathcal{O}_n^0$ と書く
- 古典分布 $p$ の対角埋め込みは $ho_p=\sum_xp(x)|x\rangle\langle x|$ と書く

## 範囲

- 本編は量子状態，量子 coupling，量子チャネル，多体系の量子 $W_1$，古典系への還元の5章とする
- 量子 $W_1$ が量子状態上の距離であることを証明する
- 対角状態上で古典 Hamming--Wasserstein 距離と一致することを証明する
- エントロピー正則化，量子 Sinkhorn，輸送不等式，半古典極限，Carlen--Maas の動的定式化は扱わない
- coupling とコスト作用素だけでは一般に距離にならないことを明記し，複数の量子 Wasserstein 理論を同一視しない

## サイト

- tex が source of truth。`make quantum-ot-site` でサイトを生成する（生成物は追跡外）
- 変換エンジンは `tools/site/` を共有し，固有設定だけを `site.config.mjs` に置く
