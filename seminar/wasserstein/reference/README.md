# Wasserstein セミナーの参考資料

主たる文献は Cédric Villani, *Optimal Transport: Old and New*, Springer, 2009
（[正式版の書誌情報](https://doi.org/10.1007/978-3-540-71050-9)）。

## 北別府先生の測度分解定理と Radon–Nikodym の定理の資料

`北別府先生/` の提供写真は，北別府悠『最適輸送理論入門〜兼非標準的関数解析入門〜』
Theorem 5.3.1 とその証明，および §4.6 の Radon–Nikodym の定理に対応する
（著者名・表題・定理番号・以下のページ番号は写真で確認）．

- `IMG_3490.JPG`：測度分解の定義と Theorem 5.3.1（存在・本質的一意性）．
- `IMG_3491.JPG`：compact な場合の構成，Radon–Nikodym 微分，可算稠密部分空間．
- `IMG_3492.JPG`：可算個の条件を共通の零集合の外で成立させる議論（印刷 p.77）．
- `IMG_3493.JPG`：Riesz–Markov–Kakutani の表現定理，可測性・積分表示・一意性，一般の場合への拡張．
- `IMG_3496.JPG`：Lebesgue 分解定理（Theorem 4.6.1）と証明の前半（印刷 p.56）．
- `IMG_3497.JPG`：同証明の後半，絶対連続性の定義，Radon–Nikodym の定理
  （Corollary 4.6.2）と証明，符号付きの場合への拡張（Corollary 4.6.3）の主張と証明の冒頭（印刷 p.57）．

本文の「Wasserstein 距離の距離性」では，接着補題で使う積空間上の形に限定し，
可測な確率測度の族と矩形上の積分公式の存在を，一つの定理として述べた．
一般の写像による形，逆像への集中，本質的一意性，一般の関数の積分公式は扱わない．
証明は資料の構成方法に沿い，第1の空間が compact な場合に Radon–Nikodym 微分と
可算な稠密部分空間から Riesz の表現定理で測度を作る．一般の Polish 空間では
第1周辺分布の tightness を用い，各制限測度の第2周辺分布の
Radon–Nikodym 微分を重みとして分解を足し合わせる．

付録では，p.57 の ε–δ による絶対連続性の定義と，Corollary 4.6.2 の三条件の同値性を記載した．
符号付きの場合は Corollary 4.6.3 のうち有限符号付き測度への適用を述べ，
本文の測度分解の証明ではこの形を使う．両者を一つの定理にまとめ，密度の一意性の議論も補った．
Corollary 4.6.3 の主張は `IMG_3497.JPG` で照合済みであり，原文の基準測度 μ と集合関数 ψ を，
付録の ν と α に対応させている．有限符号付き測度 α は有限加法的かつ有限値であり，
原文と同じ ε–δ による絶対連続性の仮定の下で適用できる．
Radon–Nikodym の定理の存在・同値性は前提とし，その証明は追加していない．
資料の Theorem 4.6.1 が有限または σ-有限の基準測度を扱うのに対し，
ここで述べる Radon–Nikodym の定理は，Corollary 4.6.2・4.6.3 と同じ有限の場合である．

前提とする Radon–Nikodym の定理と Riesz の表現定理は，
Rudin *Real and Complex Analysis* も参照．両定理の主張は
[Arun Ram の講義ノート](https://math.soimeme.org/~arunram/Notes/RadonNikodymContent.html)
でも確認できる．

## Villani の公開草稿

- ファイル：[villani-old-and-new-draft-2006-09-27.pdf](villani-old-and-new-draft-2006-09-27.pdf)
- 著者：Cédric Villani
- 書名：*Optimal transport, old and new*
- 版：2006年9月27日の出版前草稿（表紙・冒頭の説明で確認）
- 入手元：[CEREMADE に掲載された PDF](https://www.ceremade.dauphine.fr/~mischler/articles/VBook-O%26N.pdf)
- 取得日：2026年10月7日
- PDF の総ページ数：547（本文の印刷ページ番号とは異なる）
- SHA-256：`80d725bfa2da93f5517d35fab4e35b90a726973559444f6886a8ce9beef63c80`

2005年の Saint-Flour 講義ノートを改訂した原稿で、冒頭には一部の章の修正や
参考文献の補完が必要と記されている。2009年の正式版とは本文・番号・ページが異なる。
例えば Chapter 6 “The Wasserstein distances” の開始ページは、草稿では p.73、
正式版では p.93 である。

## 利用と引用の方針

- 草稿は全文検索・通読・関連箇所の探索に利用する。
- セミナーで用いる定義・記法・証明の基準は正式版（2009）とする。
- **定理・補題・定義・式・章節の番号、および引用ページ番号は正式版のものを使う。**
- 草稿で見つけた主張は、正式版の本文と照合してから引用する。対応する番号を推測しない。
- 正式版で確認できていない対応は未確認と明示する。
- 草稿内で検索語や公式が見つからなくても、正式版に記載がないとは判断しない。

正式版の保存済み写真は `old_and_new/pNNN.jpg` にあり、`NNN` は正式版の印刷ページ番号を表す。
写真は Git 追跡対象外である。
