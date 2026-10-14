## seminar/wasserstein

Wasserstein 距離の定義と距離性を扱う。Villani (2009) を主たる文献とし、
最適 coupling の存在定理を経て $W_p$ が距離であることを示す。

## 参考文献

- [Optimal Transport: Old and New, C. Villani (2009)](https://doi.org/10.1007/978-3-540-71050-9) — **主たる文献**。定義・記法・証明はこれに従う
- [最適輸送理論とリッチ曲率, 桑江ほか (Encounter with Mathematics 第63回, 2015)](https://www.math.chuo-u.ac.jp/ENCwMATH/EwM63resume.pdf) — 日本語の解説、定義の補完（§1.4 が Wasserstein 距離）

参照 PDF は `reference/` にある（kuwae-etal-2015.pdf、Billingsley-2eme-edition.pdf［Prokhorov の出典 Th 5.1。著作権のため git 追跡対象外］）。
Villani の出版前草稿（2006年9月27日版）は `reference/villani-old-and-new-draft-2006-09-27.pdf` に保存してある。全文検索・通読・関連箇所の探索に利用する。入手元と版情報は `reference/README.md` を参照。
**定理・補題・定義・式・章節の番号、および引用ページ番号は、必ず正式版（2009）で確認したものを使う。** 草稿の番号やページを正式版のものとして引用しない。草稿で見つけた主張は、正式版の該当本文と照合してから引用する。正式版で未確認の対応は推測で補わず、未確認と明示する。草稿の検索結果だけで、正式版に記載がないとは判断しない。
Villani の該当ページのスキャンは `reference/old_and_new/pNNN.jpg` にある（**ファイル名＝印刷ページ番号**）。距離性までに対応するのは pp.11–12, 43–46, 49, 93–111 であり，ほかに Definition 1.1 の切り抜き `def1-1_coupling.jpg` がある。著作権のため git 追跡対象外。追加時も同じ命名にする。

## 記法

- **記号は原則として Villani (2009) に合わせる**。ただし記号の濫用は採用しない（例: Villani が積分域 $\mathcal{X}\times\mathcal{X}$ を $\int_{\mathcal{X}}$ と略す箇所は、正確に $\int_{\mathcal{X}\times\mathcal{X}}$ と書く）
- 文献間で定義が異なる場合は、最も一般的な定義を採用し、差異を注意として記載する
- 押し出し記法 $T_\sharp\mu$ は使わず、像測度 $\mu\circ T^{-1}$ と書く（定義箇所に文献との対応を注記済み）
- 座標射影を $\mathrm{pr}_1,\mathrm{pr}_{12}$ などと記号化しない。周辺分布または座標成分を集合上の等式で直接書く
- 空間記号は付録も含め Villani に合わせて calligraphy（$\mathcal{X},\mathcal{Y},\mathcal{Z}$）で書く。例外は抽象的な測度空間 $\Omega,\Omega'$ と、距離の公理の台集合 $M$（$P_p(\mathcal{X})$ 等にも適用するため中立の記号を維持）
- 弱収束は `\weakto`（$\rightharpoonup$）で書く。Villani / Billingsley は $\Rightarrow$ を使うが、含意と紛らわしいため採用しない（$\Rightarrow$ は含意専用）
- 定着した名称のない補題は、内容を説明する日本語タイトルを付ける
- Villani に対応する番号のあるブロックは、タイトルに `（Villani Thm 4.1）` の形で併記する（番号のない主張・本資料独自の補題には付けない）
- 記法の初出箇所には「記法 …」の注意を置く（例: $L_n\downarrow L^*$、$\norm{f}_{L^p(\mu)}$）

## 方針

- 前提知識は付録（`tex/foundations/`）で全網羅し、本文からは参照でリンクする
- $W_p$ は Villani (Definition 6.1) に合わせて $1\leq p<\infty$ のみ扱う（$W_\infty$ と $L^\infty$ 系の道具は導入しない）
- 最適 coupling の存在定理は Villani (2009) Theorem 4.1 をコスト $d^p$ に特殊化した形（Remark 4.2 にいう $a=b=0$ の場合）で示し、距離性（三角不等式・一致の公理）の証明に用いる。Lemma 4.3 も「$p$ 次輸送コストの下半連続性」に特殊化（$d^p$ は連続なので切り捨て $\min\{d^p,m\}$ ＋単調収束定理で自己完結。一般コスト・上半連続 $h$・$L^1$ 条件は置かない）。Lemma 4.4（輸送計画の tightness）は Villani どおり
- Prokhorov の定理は証明なしで認める（ステートメントと引用元のみ明示。Villani Ch.4 / Billingsley 1999 Th 5.1, 5.2）
- 証明なしで認める標準定理はステートメントを明示して文献を挙げる

## サイト

- tex が source of truth。`make wasserstein-site` でサイトを生成する（生成物は追跡外）
- 定理番号は tex2md が章・節単位で振っている
- 変換エンジンは `tools/site/`（全セミナー共通）。このセミナー固有の設定は `site.config.mjs`
