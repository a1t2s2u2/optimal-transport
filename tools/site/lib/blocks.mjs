// ブロック種別の一覧。tex の定理環境・サイトの CSS・依存グラフの凡例を
// 1 か所で対応づける。ここに足せば三者が同時に追随する。

export const BLOCK_TYPES = {
  definition: { label: "定義", cssClass: "block--def", color: "var(--teal)" },
  theorem: { label: "定理", cssClass: "block--thm", color: "var(--indigo)" },
  proposition: { label: "命題", cssClass: "block--prop", color: "var(--orange)" },
  lemma: { label: "補題", cssClass: "block--lem", color: "var(--violet)" },
  claim: { label: "主張", cssClass: "block--clm", color: "var(--green)" },
  corollary: { label: "系", cssClass: "block--cor", color: "var(--cyan)" },
  algorithm: { label: "アルゴリズム", cssClass: "block--algo", color: "var(--amber)" },
  remark: { label: "注意", cssClass: null, color: "var(--muted)" },
  example: { label: "例", cssClass: null, color: "var(--wine)" },
};

// HTML の id に使う接頭辞（def-…, thm-… など）。
export const ID_PREFIX = {
  definition: "def",
  theorem: "thm",
  proposition: "prop",
  lemma: "lem",
  claim: "clm",
  corollary: "cor",
  algorithm: "algo",
  remark: "rem",
  example: "ex",
};
