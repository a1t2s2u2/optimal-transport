## seminar/wasserstein

Wasserstein 距離の理論と性質。Villani (2009) を主たる文献とし、これに従って構成する。
中心目標は、最適 coupling の存在定理を経た $W_p$ の距離性。
その後、Gaussian 測度の集中と潜在空間の最適輸送幾何への応用を扱う。

## 参考文献

- [Optimal Transport: Old and New, C. Villani (2009)](https://doi.org/10.1007/978-3-540-71050-9) — **主たる文献**。定義・記法・証明はこれに従う
- Topics in Optimal Transportation, C. Villani (2003, GSM 58, AMS) — 補助文献（1次元閉形式 Thm 2.18 の出典）
- [Optimal Latent Transport, Roy & Hauberg (NeurIPS 2022 WS)](https://openreview.net/pdf?id=ZxRYWTerLVq) — 応用章（第4章）の対象論文
- [最適輸送理論とリッチ曲率, 桑江ほか (Encounter with Mathematics 第63回, 2015)](https://www.math.chuo-u.ac.jp/ENCwMATH/EwM63resume.pdf) — 日本語の解説、定義の補完（§1.4 が Wasserstein 距離）

参照 PDF は `reference/` にある（kuwae-etal-2015.pdf、Billingsley-2eme-edition.pdf［Prokhorov の出典 Th 5.1。著作権のため git 追跡対象外］）。
Villani の該当ページのスキャンは `reference/old_and_new/pNNN.jpg` にある（**ファイル名＝ページ番号**。現在 pp.43–46, 49, 93–111, 379–383, 567–600、ほかに Definition 1.1 の切り抜き `def1-1_coupling.jpg`。著作権のため git 追跡対象外。追加時も同じ命名にする）。

## 記法

- **記号は原則として Villani (2009) に合わせる**。ただし記号の濫用は採用しない（例: Villani が積分域 $\mathcal{X}\times\mathcal{X}$ を $\int_{\mathcal{X}}$ と略す箇所は、正確に $\int_{\mathcal{X}\times\mathcal{X}}$ と書く）
- 文献間で定義が異なる場合は、最も一般的な定義を採用し、差異を注意として記載する
- 押し出し記法 $T_\sharp\mu$ は使わず、像測度 $\mu\circ T^{-1}$ と書く（定義箇所に文献との対応を注記済み）
- 空間記号は付録も含め Villani に合わせて calligraphy（$\mathcal{X},\mathcal{Y},\mathcal{Z}$）で書く。例外は抽象的な測度空間 $\Omega,\Omega'$ と、距離の公理の台集合 $M$（$P_p(\mathcal{X})$ 等にも適用するため中立の記号を維持）
- 定着した名称のない補題は、内容を説明する日本語タイトルを付ける
- Villani に対応する番号のあるブロックは、タイトルに `（Villani Thm 4.1）` の形で併記する（番号のない主張・本資料独自の補題には付けない）
- 記法の初出箇所には「記法 …」の注意を置く（例: $L_n\downarrow L^*$、$\norm{f}_{L^p(\mu)}$）

## 方針

- 前提知識は付録（`tex/foundations/`）で全網羅し、本文からは参照でリンクする
- $W_p$ は Villani (Definition 6.1) に合わせて $1\leq p<\infty$ のみ扱う（$W_\infty$ と $L^\infty$ 系の道具は導入しない）
- 最適 coupling の存在定理は Villani (2009) Theorem 4.1 をコスト $d^p$ に特殊化した形（Remark 4.2 にいう $a=b=0$ の場合）で示し、距離性（三角不等式・一致の公理）の証明に用いる。Lemma 4.3 も「$p$ 次輸送コストの下半連続性」に特殊化（$d^p$ は連続なので切り捨て $\min\{d^p,m\}$ ＋単調収束定理で自己完結。一般コスト・上半連続 $h$・$L^1$ 条件は置かない）。Lemma 4.4（輸送計画の tightness）は Villani どおり
- Prokhorov の定理は証明なしで認める（ステートメントと引用元のみ明示。Villani Ch.4 / Billingsley 1999 Th 5.1, 5.2）
- 集中の章（第3章）は Villani Ch.22 前半準拠。相対エントロピー・$T_p$ 不等式（Def 22.1）を定義し、Marton の方法（Th 22.10 の (i)⇒(iii)⇒(vi) 部分）で $T_p$ ⇒ 集中を証明。**Talagrand の不等式（標準 Gaussian は $T_2(1)$、Talagrand 1996 / Villani Ex 22.15）は admitted**。$T_p$ の双対定式化（Prop 22.3、Th 5.26 依存）と tensorization（Prop 22.5、Cor 5.22 依存）は扱わない
- 応用の章（第4章）は例外的に論文 Roy–Hauberg (2022) を対象とし、証明は本編の道具で自給する（Villani 準拠の例外）。1次元閉形式のみ Villani (2003) Thm 2.18 を admitted。同論文の式 (6) の精密化（直積分解の正確な条件）を含む
- 証明なしで認める標準定理はステートメントを明示して文献を挙げる

## サイト

- tex が source of truth。`make wasserstein-site` で content/md → dist/html を生成（生成物は追跡外）
- **PDF は生成しない**（`wasserstein-pdf` ターゲットは 2026-07-08 に削除。tex は site の source としてのみ使う）
- 定理番号は tex2md が LaTeX のカウンタ規則を再現して振っている
