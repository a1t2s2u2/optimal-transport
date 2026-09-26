// セミナーごとの site.config.mjs を読み込み、既定値を埋めて検証する。
//
// エンジン（tools/site/）はセミナーに依存しない。セミナー固有の情報は
// すべて <seminarDir>/site.config.mjs に置き、ここで一元的に解決する。

import path from "node:path";
import { existsSync } from "node:fs";
import { pathToFileURL } from "node:url";

// 章に必須のキー。
const CHAPTER_KEYS = ["tex", "md", "id", "nav", "title"];

// tex の環境名 → [サイト側コンテナ, 見出し接頭辞]。
// 接頭辞は見出しに「Def 2.1.3: …」の形で出る。
// コンテナ名はそのままブロック種別になり、配色に使う。
const DEFAULT_BLOCK_ENVS = {
  definition: ["definition", "Def"],
  claim: ["claim", "Clm"],
  lemma: ["lemma", "Lem"],
  theorem: ["theorem", "Thm"],
  proposition: ["proposition", "Prop"],
  corollary: ["corollary", "Cor"],
  remark: ["fact", "Rem"],
  example: ["fact accent", "Ex"],
};

// tex の環境名 → \label の接頭辞（\ref 解決と定理番号の引き当てに使う）。
const DEFAULT_ENV_TO_PREFIX = {
  definition: "def",
  claim: "clm",
  lemma: "lem",
  theorem: "thm",
  proposition: "prop",
  corollary: "cor",
  remark: "rem",
  example: "ex",
};

// \label の接頭辞 → 表示略号。
const DEFAULT_LABEL_PREFIX_MAP = {
  def: "Def",
  clm: "Clm",
  lem: "Lem",
  thm: "Thm",
  prop: "Prop",
  cor: "Cor",
  rem: "Rem",
  ex: "Ex",
};

// 本文中の参照語（日本語）→ 表示略号。tex 側の「定理~\ref{...}」を解釈する。
const DEFAULT_JP_TO_ABBREV = {
  定義: "Def",
  主張: "Clm",
  命題: "Prop",
  定理: "Thm",
  例: "Ex",
  注意: "Rem",
  補題: "Lem",
  系: "Cor",
  Claim: "Clm",
};

function fail(message) {
  throw new Error(`site.config.mjs: ${message}`);
}

// seminarDir（例 seminar/cuturi）から設定を読み、既定値を補って返す。
export async function loadConfig(seminarDir) {
  const dir = path.resolve(seminarDir);
  const configPath = path.join(dir, "site.config.mjs");
  if (!existsSync(configPath)) {
    throw new Error(`設定ファイルが見つからない: ${configPath}`);
  }

  const mod = await import(pathToFileURL(configPath).href);
  const raw = mod.default;
  if (!raw || typeof raw !== "object") {
    fail("default export がオブジェクトではない");
  }

  if (!raw.title) fail("title は必須");
  if (!Array.isArray(raw.chapters) || raw.chapters.length === 0) {
    fail("chapters は 1 件以上の配列でなければならない");
  }

  const seenIds = new Set();
  raw.chapters.forEach((ch, i) => {
    for (const key of CHAPTER_KEYS) {
      if (!ch[key]) fail(`chapters[${i}] に "${key}" がない`);
    }
    if (ch.group && ch.group !== "main" && ch.group !== "appendix") {
      fail(`chapters[${i}].group は "main" か "appendix" のみ`);
    }
    if (seenIds.has(ch.id)) fail(`chapters[${i}].id "${ch.id}" が重複している`);
    seenIds.add(ch.id);
  });

  const texDir = path.join(dir, "tex");
  const siteDir = path.join(dir, "site");

  return {
    // --- パス ---
    texDir,
    preamblePath: path.join(texDir, "preamble.tex"),
    contentDir: path.join(siteDir, "content"),
    distDir: path.join(siteDir, "dist"),
    texSubdirs: ["main", "foundations"],

    // --- 表示 ---
    title: raw.title,
    logo: raw.logo ?? "OT",
    siteName: raw.siteName ?? `${raw.title}セミナー`,
    landingTitle: raw.landingTitle ?? raw.title,
    landingSubtitle: raw.landingSubtitle ?? "",
    landingFooter: raw.landingFooter ?? "",
    appendixHeading: raw.appendixHeading ?? "付録：前提知識",
    appendixSubheading:
      raw.appendixSubheading ?? "本編から参照される数学的前提をまとめたもの．",
    lang: raw.lang ?? "ja",

    // --- 章立て ---
    chapters: raw.chapters.map((ch) => ({ group: "main", eyebrow: "", ...ch })),

    // --- 変換規則 ---
    blockEnvs: DEFAULT_BLOCK_ENVS,
    envToPrefix: DEFAULT_ENV_TO_PREFIX,
    labelPrefixMap: DEFAULT_LABEL_PREFIX_MAP,
    jpToAbbrev: DEFAULT_JP_TO_ABBREV,
    // --- 数式マクロ（preamble.tex から自動抽出したものへの上書き）---
    macroOverrides: raw.macroOverrides ?? {},

    // --- デモ図 ---
    demos: raw.demos ?? {},
  };
}
