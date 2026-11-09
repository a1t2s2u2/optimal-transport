#!/usr/bin/env node
// tex の記法と文章構成を検査する。
//
//   node tools/notation/lint.mjs <セミナーのディレクトリ>
//
// 違反を「ファイル:行: [規則] 内容」の形で出し、1 件でもあれば終了コード 1 で終わる。
// 規則は lib/lint.mjs、セミナー固有の禁止記法は site.config.mjs の notation にある。

import path from "node:path";
import { loadConfig } from "../site/lib/config.mjs";
import { lintSeminar } from "./lib/lint.mjs";

const [seminarDir] = process.argv.slice(2);
if (!seminarDir) {
  console.error("usage: node tools/notation/lint.mjs <セミナーのディレクトリ>");
  process.exit(2);
}

const violations = lintSeminar(await loadConfig(seminarDir));
for (const { file, line, rule, message } of violations) {
  console.log(`${path.relative(process.cwd(), file)}:${line}: [${rule}] ${message}`);
}
console.log(violations.length ? `\n${violations.length} 件の違反` : "記法の検査: 違反なし");
process.exit(violations.length ? 1 : 0);
