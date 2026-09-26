// ブロック種別の一覧。tex の定理環境とサイト表示を対応づける。

export const BLOCK_TYPES = {
  definition: { label: "定義", cssClass: "block--def" },
  theorem: { label: "定理", cssClass: "block--thm" },
  proposition: { label: "命題", cssClass: "block--prop" },
  lemma: { label: "補題", cssClass: "block--lem" },
  claim: { label: "主張", cssClass: "block--clm" },
  corollary: { label: "系", cssClass: "block--cor" },
  remark: { label: "注意", cssClass: null },
  example: { label: "例", cssClass: null },
};

// HTML の id に使う接頭辞（def-…, thm-… など）。
export const ID_PREFIX = {
  definition: "def",
  theorem: "thm",
  proposition: "prop",
  lemma: "lem",
  claim: "clm",
  corollary: "cor",
  remark: "rem",
  example: "ex",
};
