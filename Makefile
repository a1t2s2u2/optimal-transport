# 最適輸送セミナー
#
# source of truth は各 tex/*.tex のみ。
# Web 版（site/）の md/html は生成物であり、編集も git 管理もしない。

.PHONY: cuturi-site cuturi-pdf wasserstein-site

# --- Cuturi ---
cuturi-site:
	node seminar/cuturi/site/scripts/tex2md.mjs
	node seminar/cuturi/site/scripts/build.mjs
	@echo "→ seminar/cuturi/site/dist/index.html をブラウザで開いてください"

cuturi-pdf:
	cd seminar/cuturi/tex && latexmk

# --- Wasserstein ---
# PDF は生成しない（site のみ）。tex は site の source としてのみ使う。
wasserstein-site:
	node seminar/wasserstein/site/scripts/tex2md.mjs
	node seminar/wasserstein/site/scripts/build.mjs
	@echo "→ seminar/wasserstein/site/dist/index.html をブラウザで開いてください"
