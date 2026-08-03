# 最適輸送セミナー
#
# source of truth は各 seminar/*/tex/*.tex のみ。
# Web 版（site/）の md/html は生成物であり、編集も git 管理もしない。
#
# サイト生成のエンジンは tools/site/ にあり、全セミナーで共有する。
# セミナー固有の設定は seminar/<名前>/site.config.mjs に置く。

SITE := node tools/site

.PHONY: help sites cuturi-site cuturi-pdf wasserstein-site clean-sites paper paper-ja paper-experiments paper-visualization-experiment paper-surface-experiment paper-convex-experiment paper-cartography-experiment paper-distillation-experiment paper-mnist-experiment paper-geodesic-experiment paper-architecture-figure paper-all

help:
	@echo "make sites             すべてのセミナーのサイトを生成"
	@echo "make cuturi-site       計算最適輸送のサイトを生成"
	@echo "make cuturi-pdf        計算最適輸送の PDF を生成"
	@echo "make wasserstein-site  Wasserstein 距離のサイトを生成（PDF は生成しない）"
	@echo "make paper-experiments 監査可能なWasserstein可視化実験を再実行"
	@echo "make paper-visualization-experiment 現論文の可視化実験だけを再実行"
	@echo "make paper-surface-experiment 曲面復元実験だけを再実行"
	@echo "make paper-convex-experiment 同一骨格の凸面実現実験だけを再実行"
	@echo "make paper-distillation-experiment 幾何保存蒸留の容量実験を再実行"
	@echo "make paper-mnist-experiment MNIST/FashionMNIST VAE head圧縮実験を再実行"
	@echo "make paper-geodesic-experiment 潜在直線・数値測地線比較を再実行"
	@echo "make paper             英語論文の PDF を生成"
	@echo "make paper-ja          日本語論文の PDF を生成"
	@echo "make paper-all         数値実験を再実行して両言語の PDF を生成"
	@echo "make clean-sites       生成したサイトを削除"

sites: cuturi-site wasserstein-site

# --- Cuturi ---
cuturi-site:
	$(SITE)/tex2md.mjs seminar/cuturi
	$(SITE)/build.mjs seminar/cuturi
	@echo "→ seminar/cuturi/site/dist/index.html をブラウザで開いてください"

cuturi-pdf:
	cd seminar/cuturi/tex && latexmk

# --- Wasserstein ---
# PDF は生成しない（site のみ）。tex は site の source としてのみ使う。
wasserstein-site:
	$(SITE)/tex2md.mjs seminar/wasserstein
	$(SITE)/build.mjs seminar/wasserstein
	@echo "→ seminar/wasserstein/site/dist/index.html をブラウザで開いてください"

# --- 論文 ---
# セミナー資料とは独立。既知の内容は引用で済ませ、新規の主張だけを書く。
paper-cartography-experiment:
	uv run --python 3.12 paper/ot-manifold-approximation/experiments/wasserstein_cartography.py

paper-visualization-experiment:
	uv run --python 3.12 paper/ot-manifold-approximation/experiments/ot_isometric_visualization.py

paper-surface-experiment:
	uv run --python 3.12 paper/ot-manifold-approximation/experiments/wasserstein_surface_reconstruction.py

paper-convex-experiment:
	uv run --python 3.12 paper/ot-manifold-approximation/experiments/convex_edge_realization_prototype.py

paper-distillation-experiment:
	uv run --python 3.12 paper/ot-manifold-approximation/experiments/geometry_distillation.py
	uv run --python 3.12 paper/ot-manifold-approximation/experiments/lowrank_torus.py

paper-mnist-experiment:
	uv run --python 3.12 paper/ot-manifold-approximation/experiments/mnist_low_rank_geometry.py --dataset mnist
	uv run --python 3.12 paper/ot-manifold-approximation/experiments/mnist_low_rank_geometry.py --dataset fashion-mnist

paper-geodesic-experiment:
	uv run --python 3.12 paper/ot-manifold-approximation/experiments/geodesic_interpolation.py --dataset mnist --student-ranks 8 12
	uv run --python 3.12 paper/ot-manifold-approximation/experiments/geodesic_interpolation.py --dataset fashion-mnist --student-ranks 8 13

paper-architecture-figure:
	uv run --python 3.12 paper/ot-manifold-approximation/experiments/certified_distillation_architecture.py

paper-experiments: paper-visualization-experiment

paper:
	cd paper/ot-manifold-approximation && latexmk
	@echo "→ paper/ot-manifold-approximation/out/main.pdf"

paper-ja:
	cd paper/ot-manifold-approximation && latexmk -lualatex main-ja.tex
	@echo "→ paper/ot-manifold-approximation/out/main-ja.pdf"

paper-all: paper-experiments paper paper-ja

clean-sites:
	rm -rf seminar/*/site
