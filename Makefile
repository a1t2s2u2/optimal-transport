# 最適輸送セミナー
#
# source of truth は各 seminar/*/tex/*.tex のみ。
# Web 版（site/）の md/html は生成物であり、編集も git 管理もしない。
#
# サイト生成のエンジンは tools/site/ にあり、全セミナーで共有する。
# セミナー固有の設定は seminar/<名前>/site.config.mjs に置く。

SITE := node tools/site

.PHONY: help sites cuturi-site cuturi-pdf wasserstein-site clean-sites paper paper-experiments paper-all

help:
	@echo "make sites             すべてのセミナーのサイトを生成"
	@echo "make cuturi-site       計算最適輸送のサイトを生成"
	@echo "make cuturi-pdf        計算最適輸送の PDF を生成"
	@echo "make wasserstein-site  Wasserstein 距離のサイトを生成（PDF は生成しない）"
	@echo "make paper-experiments 多視点画像の潜在多様体選択実験を再実行"
	@echo "make paper             論文の PDF を生成"
	@echo "make paper-all         数値実験を再実行して論文の PDF を生成"
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
paper-experiments:
	uv run --python 3.12 paper/ot-manifold-approximation/experiments/rotation_image_manifold.py

paper:
	cd paper/ot-manifold-approximation && latexmk
	@echo "→ paper/ot-manifold-approximation/out/main.pdf"

paper-all: paper-experiments paper

clean-sites:
	rm -rf seminar/*/site
