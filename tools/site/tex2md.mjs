#!/usr/bin/env node
// tex/*.tex → site/content/*.md
//
//   node tools/site/tex2md.mjs <セミナーのディレクトリ> [--strict]
//
// 例: node tools/site/tex2md.mjs seminar/cuturi
//
// --strict を付けると lint 警告（未解決の \ref・未変換マクロ）で失敗する。CI 用。

import { loadConfig } from "./lib/config.mjs";
import { Converter } from "./lib/tex2md.mjs";

const args = process.argv.slice(2);
const strict = args.includes("--strict") || process.env.TEX2MD_STRICT === "1";
const seminarDir = args.find((a) => !a.startsWith("--"));

if (!seminarDir) {
  console.error("usage: node tools/site/tex2md.mjs <セミナーのディレクトリ> [--strict]");
  process.exit(2);
}

const config = await loadConfig(seminarDir);
const converter = new Converter(config);
const { warnings } = converter.run();

if (warnings.length) {
  console.warn(`\n${warnings.length} lint warning(s):`);
  for (const w of warnings) {
    const loc = w.line ? `${w.file}:${w.line}` : w.file;
    console.warn(`  [${w.kind}] ${loc}  ${w.detail ?? ""}`.trimEnd());
  }
  if (strict) {
    console.error(`\nFAILED: ${warnings.length} lint warning(s) under --strict.`);
    process.exit(1);
  }
} else {
  console.log("\nLint: no warnings.");
}
