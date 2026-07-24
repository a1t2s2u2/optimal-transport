// セミナーごとの site.config.mjs を読み込み、既定値を埋めて検証する。
//
// エンジン（tools/site/）はセミナーに依存しない。セミナー固有の情報は
// すべて <seminarDir>/site.config.mjs に置き、ここで一元的に解決する。

import path from "node:path";
import { existsSync } from "node:fs";
import { pathToFileURL } from "node:url";

// 章に必須のキー。
const CHAPTER_KEYS = ["tex", "md", "id", "nav", "title"];

// UI 機能のフラグ。既定は「全部入り」。
// site.config.mjs の features で個別に false にできる。
const DEFAULT_FEATURES = {
  chapterStats: true, // ランディングの章カードに「N 定義・M 定理」を出す
  heroDecoration: true, // ランディング見出し背景の装飾 SVG
  tocProgress: true, // 章内目次の現在位置インジケータ
  fadeIn: true, // ブロックのスクロール・フェードイン
  refPulse: true, // 参照クリック時に本文側ブロックを光らせる
  keyboardHelp: true, // "?" でショートカット一覧を表示
};

// tex の環境名 → [サイト側コンテナ, 見出し接頭辞]。
// 接頭辞は見出しに「Def 2.1.3: …」の形で出る。
// コンテナ名はそのままブロック種別になり、配色・依存グラフの絞り込みに効く
// （対応表は lib/blocks.mjs）。tex 側の環境と 1 対 1 に保つこと。
const DEFAULT_BLOCK_ENVS = {
  definition: ["definition", "Def"],
  claim: ["claim", "Clm"],
  lemma: ["lemma", "Lem"],
  theorem: ["theorem", "Thm"],
  proposition: ["proposition", "Prop"],
  corollary: ["corollary", "Cor"],
  remark: ["fact", "Rem"],
  example: ["fact accent", "Ex"],
  algorithm: ["algorithm", ""],
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

// 依存グラフに載せるブロック種別。証明を伴う主張はすべて載せる
// （注意・例はグラフの本筋ではないので除く）。
const DEFAULT_GRAPH_TYPES = [
  "definition",
  "theorem",
  "proposition",
  "lemma",
  "claim",
  "corollary",
];

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

  const texDir = path.join(dir, raw.texDir ?? "tex");
  const siteDir = path.join(dir, raw.siteDir ?? "site");

  return {
    // --- パス ---
    seminarDir: dir,
    texDir,
    preamblePath: path.join(texDir, raw.preamble ?? "preamble.tex"),
    contentDir: path.join(siteDir, "content"),
    distDir: path.join(siteDir, "dist"),
    // tex の走査対象サブディレクトリ（ラベル・章タイトルの解決に使う）
    texSubdirs: raw.texSubdirs ?? ["main", "foundations"],

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
    blockEnvs: { ...DEFAULT_BLOCK_ENVS, ...raw.blockEnvs },
    envToPrefix: { ...DEFAULT_ENV_TO_PREFIX, ...raw.envToPrefix },
    labelPrefixMap: { ...DEFAULT_LABEL_PREFIX_MAP, ...raw.labelPrefixMap },
    jpToAbbrev: { ...DEFAULT_JP_TO_ABBREV, ...raw.jpToAbbrev },
    graphTypes: new Set(raw.graphTypes ?? DEFAULT_GRAPH_TYPES),

    // --- 数式マクロ（preamble.tex から自動抽出したものへの追加・除外）---
    macroOverrides: raw.macroOverrides ?? {},
    macroIgnore: raw.macroIgnore ?? [],

    // --- 用語集・デモ図 ---
    glossary: raw.glossary ?? {},
    demos: raw.demos ?? {},

    // --- 機能フラグ ---
    features: { ...DEFAULT_FEATURES, ...raw.features },
  };
}
