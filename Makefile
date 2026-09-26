# 最適輸送セミナー
#
# source of truth は各 seminar/*/tex/*.tex のみ。
# Web 版（site/）の md/html は生成物であり、編集も git 管理もしない。
#
# サイト生成のエンジンは tools/site/ にあり、全セミナーで共有する。
# セミナー固有の設定は seminar/<名前>/site.config.mjs に置く。

SITE := node tools/site

.PHONY: help sites cuturi-site wasserstein-site clean-sites

help:
	@echo "make sites             すべてのセミナーのサイトを生成"
	@echo "make cuturi-site       計算最適輸送のサイトを生成"
	@echo "make wasserstein-site  Wasserstein 距離のサイトを生成"
	@echo "make clean-sites       生成したサイトを削除"

sites: cuturi-site wasserstein-site

# --- Cuturi ---
cuturi-site:
	$(SITE)/tex2md.mjs seminar/cuturi
	$(SITE)/build.mjs seminar/cuturi
	@echo "→ seminar/cuturi/site/dist/index.html をブラウザで開いてください"

# --- Wasserstein ---
wasserstein-site:
	$(SITE)/tex2md.mjs seminar/wasserstein
	$(SITE)/build.mjs seminar/wasserstein
	@echo "→ seminar/wasserstein/site/dist/index.html をブラウザで開いてください"

clean-sites:
	rm -rf seminar/*/site
