// tex の記法と文章構成を検査する。
//
// 規則は次の 4 つ。違反は { file, line, rule, message } の配列で返す。
//
//   forbidden         site.config.mjs の notation.forbidden に挙げた書き方を使っていない
//   macro-literal     preamble.tex のマクロと同じものを直書きしていない（\mathbb{R} ではなく \R）
//   definition-term   定義ブロックに定義される語（\textbf{…}）がある
//   proof-or-source   定理・補題・命題・系には証明がある。証明しないときは
//                     「（証明は省略する．〈出典〉）」の形で出典を挙げる
//
// 「% notation: ignore」と書いた行は検査しない（記法そのものに言及する行など）。
// ブロックの規則では、\begin の行に書くとそのブロックを検査しない。

import { readFileSync } from "node:fs";
import path from "node:path";
import { extractMacros } from "../../site/lib/macros.mjs";

const IGNORE_MARK = "% notation: ignore";
const PROVABLE_ENVS = ["theorem", "lemma", "proposition", "corollary"];
const OMISSION = "証明は省略する";
// 出典とみなす書き方：年 (2009)、Theorem / Thm / Corollary / Cor / Chapter / §
const CITATION = /\(\d{4}\)|Theorems?\b|Thm\b|Corollary\b|Cor\b|Chapter\b|§/;

// TeX の行コメントを落とす（\% は残す）。行数は変えない。
function stripComment(line) {
  for (let i = 0; i < line.length; i += 1) {
    if (line[i] === "%" && (i === 0 || line[i - 1] !== "\\")) return line.slice(0, i);
  }
  return line;
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// preamble のマクロから「直書きを禁止する文字列 → 使うべきマクロ」を作る。
// 引数付きマクロは最初の引数より前の部分（\abs なら \lvert）を使う。
// 1 文字の本体（\Wass = W）のように、直書きと区別できないものは除く。
function macroLiterals(preamblePath) {
  const literals = [];
  for (const [name, def] of Object.entries(extractMacros(preamblePath))) {
    const body = Array.isArray(def) ? def[0].split("#")[0].trim() : def;
    if (!body.startsWith("\\") || body.length < 3) continue;
    // \Pi が \Pisomething の一部に当たらないよう、英字で終わるときは後続の英字を禁じる。
    // \left\lvert のような大きさ指定付きの区切り記号はマクロで書けないので除く。
    const head = "(?<!\\\\(?:left|right|[bB]igg?[lr]?))";
    const tail = /[a-zA-Z]$/.test(body) ? "(?![a-zA-Z])" : "";
    literals.push({ re: new RegExp(head + escapeRegExp(body) + tail, "g"), body, macro: `\\${name}` });
  }
  return literals;
}

function lineOf(text, index) {
  return text.slice(0, index).split("\n").length;
}

// text[at] を含む全角括弧（…）の範囲。入れ子の括弧は対応を取って飛ばす。
function enclosingParens(text, at) {
  let open = at;
  for (let depth = 0; open > 0; open -= 1) {
    if (text[open - 1] === "）") depth += 1;
    else if (text[open - 1] === "（" && depth-- === 0) break;
  }
  let close = at;
  for (let depth = 0; close < text.length; close += 1) {
    if (text[close] === "（") depth += 1;
    else if (text[close] === "）" && depth-- === 0) break;
  }
  return text.slice(open - 1, close + 1);
}

// 1 行ずつ見る規則（forbidden, macro-literal）。
function lintLines(lines, forbidden, literals, report) {
  lines.forEach((raw, i) => {
    if (raw.includes(IGNORE_MARK)) return;
    const line = stripComment(raw);
    for (const { pattern, use, why } of forbidden) {
      for (const m of line.matchAll(new RegExp(pattern.source, "g"))) {
        report(i + 1, "forbidden", `「${m[0]}」は使わない．${use} を使う（${why}）`);
      }
    }
    for (const { re, body, macro } of literals) {
      if (line.match(re)) report(i + 1, "macro-literal", `「${body}」は直書きせず ${macro} を使う`);
    }
  });
}

// ブロック単位で見る規則（definition-term, proof-or-source）。
// rawLines は コメントを落とす前の行（ignore の印を見るため）。
function lintBlocks(text, rawLines, deferredProofs, envToPrefix, report) {
  const ignored = (index) => rawLines[lineOf(text, index) - 1].includes(IGNORE_MARK);
  for (const m of text.matchAll(/\\begin\{definition\}([\s\S]*?)\\end\{definition\}/g)) {
    if (ignored(m.index)) continue;
    if (!m[1].includes("\\textbf{")) {
      report(lineOf(text, m.index), "definition-term", "定義ブロックに定義される語（\\textbf{…}）がない");
    }
  }

  const envs = PROVABLE_ENVS.join("|");
  const blockRe = new RegExp(`\\\\begin\\{(${envs})\\}\\{[^\\n]*\\}\\{([^}]*)\\}([\\s\\S]*?)\\\\end\\{\\1\\}`, "g");
  for (const m of text.matchAll(blockRe)) {
    const [whole, env, label, body] = m;
    if (ignored(m.index)) continue;
    const after = text.slice(m.index + whole.length);
    if (/^\s*\\begin\{proof\}/.test(after)) continue;
    if (deferredProofs.has(`${envToPrefix[env]}:${label}`)) continue;
    const line = lineOf(text, m.index);
    // 省略の注記はブロックの中か、直後の段落に置く
    const nearby = body + after.split(/\n\s*\n/)[0];
    const at = nearby.indexOf(OMISSION);
    if (at < 0) {
      report(line, "proof-or-source", `証明がない．証明を付けるか「（${OMISSION}．〈出典〉）」と書く`);
      continue;
    }
    const note = enclosingParens(nearby, at);
    if (!CITATION.test(note)) {
      report(line, "proof-or-source", `証明を省略しているが出典がない: ${note.replace(/\s+/g, " ")}`);
    }
  }
}

// セミナー全体を検査する。config は tools/site/lib/config.mjs の loadConfig の結果。
export function lintSeminar(config) {
  const violations = [];
  const literals = macroLiterals(config.preamblePath);
  const files = config.chapters.map((ch) => path.join(config.texDir, ch.tex));
  const texts = new Map(
    files.map((file) => [file, readFileSync(file, "utf-8").split("\n").map(stripComment).join("\n")]),
  );

  // 後ろの節で証明される定理：\begin{proof}[定理~\ref{thm:…}の証明]
  const deferredProofs = new Set();
  for (const text of texts.values()) {
    for (const m of text.matchAll(/\\begin\{proof\}\[[^\]]*?\\ref\{([^}]+)\}/g)) deferredProofs.add(m[1]);
  }

  for (const file of files) {
    const report = (line, rule, message) => violations.push({ file, line, rule, message });
    const rawLines = readFileSync(file, "utf-8").split("\n");
    lintLines(rawLines, config.notation.forbidden, literals, report);
    lintBlocks(texts.get(file), rawLines, deferredProofs, config.envToPrefix, report);
  }
  return violations.sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line);
}
