// preamble.tex のマクロ定義を MathJax のマクロ表へ変換する。
//
// サイトの数式は MathJax が描画するため、tex 中のユーザ定義マクロ（\R, \abs など）
// を MathJax にも教える必要がある。手で二重管理すると必ずずれるので、
// preamble.tex を唯一の定義元として機械的に抽出する。

import { readFileSync } from "node:fs";

// 対応する定義形式：
//   \newcommand{\NAME}{BODY}
//   \newcommand{\NAME}[N]{BODY}
//   \renewcommand{...} も同様
//   \DeclareMathOperator{\NAME}{TEXT}   -> \operatorname{TEXT}
//   \DeclareMathOperator*{\NAME}{TEXT}  -> \operatorname*{TEXT}
//
// 本文が空のもの（\blockmeta, \demohint のような PDF 側で無効化する指示マクロ）は
// 数式マクロではないので除く。

// start（'{' の位置）から波括弧の対応を取って中身と次位置を返す。
function extractBraceArg(text, start) {
  if (start >= text.length || text[start] !== "{") return null;
  let depth = 0;
  for (let i = start; i < text.length; i += 1) {
    if (text[i] === "\\") {
      i += 1; // \{ \} をエスケープとして読み飛ばす
      continue;
    }
    if (text[i] === "{") depth += 1;
    else if (text[i] === "}") {
      depth -= 1;
      if (depth === 0) return [text.slice(start + 1, i), i + 1];
    }
  }
  return null;
}

// TeX の行コメントを落とす（\% は残す）。
function stripComments(source) {
  return source
    .split("\n")
    .map((line) => {
      for (let i = 0; i < line.length; i += 1) {
        if (line[i] === "%" && (i === 0 || line[i - 1] !== "\\")) return line.slice(0, i);
      }
      return line;
    })
    .join("\n");
}

// preamble.tex のパスから MathJax マクロ表を作る。
// 返り値は { NAME: "BODY" } または { NAME: ["BODY", 引数の個数] }。
export function extractMacros(preamblePath, { warn = () => {} } = {}) {
  const source = stripComments(readFileSync(preamblePath, "utf-8"));
  const macros = {};

  // --- \newcommand / \renewcommand ---
  const cmdRe = /\\(?:re)?newcommand\s*\{\s*\\([a-zA-Z]+)\s*\}\s*(\[(\d+)\])?\s*(\[)?/g;
  for (const m of source.matchAll(cmdRe)) {
    const name = m[1];
    const argc = m[3] ? Number(m[3]) : 0;
    // \newcommand{\foo}[1][default]{...} の既定値付きは MathJax に写せない。
    if (m[4]) {
      warn(`\\newcommand{\\${name}} は既定値付きオプション引数のため取り込めない`);
      continue;
    }
    const bodyStart = m.index + m[0].length;
    const body = extractBraceArg(source, bodyStart);
    if (body === null) {
      warn(`\\newcommand{\\${name}} の本体を読み取れない`);
      continue;
    }
    const text = body[0].trim();
    if (text === "") continue; // 指示マクロ（\blockmeta 等）は数式マクロではない
    macros[name] = argc > 0 ? [text, argc] : text;
  }

  // --- \DeclareMathOperator ---
  const opRe = /\\DeclareMathOperator\s*(\*)?\s*\{\s*\\([a-zA-Z]+)\s*\}\s*/g;
  for (const m of source.matchAll(opRe)) {
    const starred = Boolean(m[1]);
    const name = m[2];
    const body = extractBraceArg(source, m.index + m[0].length);
    if (body === null) {
      warn(`\\DeclareMathOperator{\\${name}} の本体を読み取れない`);
      continue;
    }
    macros[name] = `\\operatorname${starred ? "*" : ""}{${body[0].trim()}}`;
  }

  return macros;
}

// 抽出したマクロに設定側の上書き・除外を適用する。
// MathJax が解釈できない綴り（stmaryrd の \llbracket など）は
// site.config.mjs の macroOverrides で差し替える。
export function resolveMacros(preamblePath, { overrides = {}, ignore = [], warn } = {}) {
  const macros = extractMacros(preamblePath, { warn });
  for (const name of ignore) delete macros[name];
  return { ...macros, ...overrides };
}

// MathJax の設定に埋め込める JS リテラルへ整形する。
export function macrosToJsLiteral(macros, indent = "            ") {
  const entries = Object.entries(macros).map(([name, value]) => {
    const body = Array.isArray(value)
      ? `[${JSON.stringify(value[0])}, ${value[1]}]`
      : JSON.stringify(value);
    return `${indent}${name}: ${body},`;
  });
  return entries.join("\n");
}
