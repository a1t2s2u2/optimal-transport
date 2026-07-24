// TeX の章ファイルを、サイト用の中間 markdown へ変換する。
//
// 出力する content/*.md の方言（frontmatter + 独自ブロック記法）は
// markdown.mjs がそのままパースする中間生成物。tex が source of truth。

import { readFileSync, writeFileSync, readdirSync, existsSync, rmSync, mkdirSync } from "node:fs";
import path from "node:path";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// TeX の行コメント（% ...）を落とす。\% は残す。
function stripComments(line) {
  const result = [];
  for (let i = 0; i < line.length; i++) {
    if (line[i] === "%" && (i === 0 || line[i - 1] !== "\\")) break;
    result.push(line[i]);
  }
  return result.join("").replace(/\s+$/, "");
}

// $...$ を \(...\) にする（$$...$$ は現れない前提）。
function convertInlineMath(text) {
  const parts = [];
  let inMath = false;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "$" && (i === 0 || text[i - 1] !== "\\")) {
      parts.push(inMath ? "\\)" : "\\(");
      inMath = !inMath;
    } else {
      parts.push(text[i]);
    }
  }
  // 閉じていない数式が残る場合は変換しない（行連結後は起きない想定の保険）
  if (inMath) return text;
  return parts.join("");
}

// \textbf{X} -> **X**、\textit{X}/\emph{X} -> *X*、\textrm{X} -> X。
// 波括弧のネストは 1 段まで扱う。
function convertTextCommands(text) {
  const nested = "(?:[^{}]|\\{[^{}]*\\})*";
  text = text.replace(new RegExp(`\\\\textbf\\{(${nested})\\}`, "g"), "**$1**");
  text = text.replace(new RegExp(`\\\\textit\\{(${nested})\\}`, "g"), "*$1*");
  text = text.replace(new RegExp(`\\\\emph\\{(${nested})\\}`, "g"), "*$1*");
  text = text.replace(new RegExp(`\\\\textrm\\{(${nested})\\}`, "g"), "$1");
  return text;
}

function stripLabel(text) {
  return text.replace(/\\label\{[^}]*\}/g, "");
}

function stripRef(text) {
  text = text.replace(/第~?\\ref\{[^}]*\}~?章/g, "");
  text = text.replace(/§~?\\ref\{[^}]*\}/g, "");
  text = text.replace(/~?\\ref\{[^}]*\}/g, "");
  text = text.replace(/  +/g, " ");
  text = text.replace(/（\s*）/g, "");
  text = text.replace(/\(\s*\)/g, "");
  return text;
}

// start（'{' の位置）から対応する '}' までを取り出す。[中身, 次位置] を返す。
function extractBraceArg(text, start) {
  if (start >= text.length || text[start] !== "{") return null;
  let depth = 0;
  for (let i = start; i < text.length; i++) {
    if (text[i] === "{") depth += 1;
    else if (text[i] === "}") {
      depth -= 1;
      if (depth === 0) return [text.slice(start + 1, i), i + 1];
    }
  }
  return null;
}

// \texorpdfstring{A}{B} -> A（見出し・本文中の表示用整形を除く）。
function convertTexorpdfstring(text) {
  const nested = "(?:[^{}]|\\{[^{}]*\\})*";
  return text.replace(
    new RegExp(`\\\\texorpdfstring\\{(${nested})\\}\\{${nested}\\}`, "g"),
    "$1",
  );
}

// ---------------------------------------------------------------------------
// Converter
// ---------------------------------------------------------------------------

// 1 セミナー分の変換器。設定を閉じ込め、モジュール全体で状態を共有しない。
class Converter {
  constructor(config) {
    this.config = config;
    this.labelMap = {};
    this.chapterMap = {};
    this.numberMap = {};
    this.warnings = [];
    this.currentFile = null;
    // 「数式の外に出ても安全」と確認できたマクロだけここに足す。既定は空。
    // 埋めると \emph 級の素通りバグを隠してしまう。
    this.macroWhitelist = new Set(config.macroWhitelist ?? []);
  }

  warn(entry) {
    this.warnings.push({ file: this.currentFile, ...entry });
  }

  // 本編・付録の全 tex を出現順に返す（preamble.tex / main.tex は含めない）。
  allChapterTex() {
    const paths = [];
    for (const sub of this.config.texSubdirs) {
      const dir = path.join(this.config.texDir, sub);
      if (!existsSync(dir)) continue;
      for (const name of readdirSync(dir)) {
        if (name.endsWith(".tex")) paths.push(path.join(dir, name));
      }
    }
    paths.sort();
    return paths;
  }

  // 全 tex を走査して 'prefix:label' -> タイトル の写像を作る。
  buildLabelMap() {
    const { envToPrefix } = this.config;
    const labelMap = {};
    for (const texPath of this.allChapterTex()) {
      const content = readFileSync(texPath, "utf-8");
      for (const m of content.matchAll(/\\begin\{(\w+)\}/g)) {
        const envName = m[1];
        if (!(envName in envToPrefix)) continue;
        let pos = m.index + m[0].length;
        const titleResult = extractBraceArg(content, pos);
        if (titleResult === null) continue;
        pos = titleResult[1];
        const labelResult = extractBraceArg(content, pos);
        if (labelResult === null) continue;
        labelMap[`${envToPrefix[envName]}:${labelResult[0]}`] = titleResult[0];
      }
    }
    return labelMap;
  }

  // PDF（LaTeX）と同一の定理番号「章.節.通し番号」を再現する。
  // 規則は preamble.tex の tcolorbox 設定に対応する：
  //   - 全定理環境が definition のカウンタを共有（use counter from）
  //   - カウンタは節ごとにリセット（number within=section）
  //   - 章番号は本編 1,2,…、付録 A,B,…（main.tex の \appendix による）
  //   - \section*（星付き）は節番号を進めない
  // 返り値は 'prefix:label' -> '2.1.4' の写像。
  buildNumberMap() {
    const { envToPrefix, chapters, texDir } = this.config;
    const map = {};
    let mainNo = 0;
    let appendixNo = 0;
    for (const ch of chapters) {
      const texPath = path.join(texDir, ch.tex);
      if (!existsSync(texPath)) continue;
      let chapterNo;
      if (ch.group === "appendix") {
        appendixNo += 1;
        chapterNo = String.fromCharCode(64 + appendixNo); // 1->A, 2->B, ...
      } else {
        mainNo += 1;
        chapterNo = String(mainNo);
      }
      const content = readFileSync(texPath, "utf-8");
      let sectionNo = 0;
      let counter = 0;
      for (const m of content.matchAll(/\\section\{|\\begin\{(\w+)\}/g)) {
        if (m[0] === "\\section{") {
          sectionNo += 1;
          counter = 0;
          continue;
        }
        const envName = m[1];
        if (!(envName in envToPrefix)) continue;
        const pos = m.index + m[0].length;
        const titleResult = extractBraceArg(content, pos);
        if (titleResult === null) continue;
        const labelResult = extractBraceArg(content, titleResult[1]);
        if (labelResult === null) continue;
        counter += 1;
        map[`${envToPrefix[envName]}:${labelResult[0]}`] =
          `${chapterNo}.${sectionNo}.${counter}`;
      }
    }
    return map;
  }

  // 全 tex から \chapter{TITLE}\label{ch:...} の対を集める。
  buildChapterMap() {
    const chapterMap = {};
    const nested = "(?:[^{}]|\\{[^{}]*\\})*";
    const texor = new RegExp(
      `\\\\texorpdfstring\\{(${nested})\\}\\{${nested}\\}`,
      "g",
    );
    for (const texPath of this.allChapterTex()) {
      const content = readFileSync(texPath, "utf-8");
      for (const m of content.matchAll(/\\chapter\{/g)) {
        const res = extractBraceArg(content, m.index + m[0].length - 1);
        if (res === null) continue;
        const [title, pos] = res;
        const labelM = content.slice(pos, pos + 200).match(/^\s*\\label\{(ch:[^}]*)\}/);
        if (!labelM) continue;
        chapterMap[labelM[1]] = title.replace(texor, "$1").trim();
      }
    }
    return chapterMap;
  }

  // \ref{...} をクリック可能な [ref:表示|参照名] に変換する。
  convertRefs(text) {
    const { labelPrefixMap, jpToAbbrev } = this.config;
    text = text.replace(/第~?\\ref\{(ch:[^}]*)\}~?章/g, (_, label) => {
      const title = this.chapterMap[label];
      return title ? `「${title}」の章` : "本章";
    });
    text = text.replace(/§~?\\ref\{sec:[^}]*\}/g, "本節");
    text = text.replace(/(?:Algorithm|アルゴリズム)~?\\ref\{alg:[^}]*\}/g, "アルゴリズム");
    // 図は tikz のためサイトでは省略される。図参照は語「図」に落として壊さない。
    text = text.replace(/図~?\\ref\{fig:[^}]*\}/g, "図");

    const jpWords = Object.keys(jpToAbbrev).map(escapeRegExp).join("|");
    const typedRefRe = new RegExp(`(${jpWords})~?\\\\ref\\{([^}]+)\\}`, "g");
    text = text.replace(typedRefRe, (_, word, label) => {
      const title = this.labelMap[label];
      if (!title) {
        this.warn({ kind: "unresolved-ref", detail: `${word}~\\ref{${label}}` });
        return "";
      }
      const prefix = label.includes(":") ? label.split(":")[0] : "";
      // 参照語（命題/補題/…）とラベル接頭辞（prop/lem/…）の食い違いを検出
      const expected = jpToAbbrev[word];
      const actual = labelPrefixMap[prefix];
      if (expected && actual && expected !== actual) {
        this.warn({
          kind: "ref-type-mismatch",
          detail: `${word}~\\ref{${label}}: 参照語「${word}」(${expected}) がラベル接頭辞「${prefix}」(${actual}) と不一致`,
        });
      }
      const abbrev = labelPrefixMap[prefix] || expected || word;
      // PDF と同じ番号があれば「補題 2.2.3」の形で短く示す（クリックで右に本文）。
      const num = this.numberMap[label];
      const display = num ? `${word} ${num}` : `${abbrev}: ${title}`;
      return `[ref:${display}|${title}]`;
    });

    text = text.replace(/~?\\ref\{([^}]+)\}/g, (_, label) => {
      const title = this.labelMap[label];
      if (!title) {
        this.warn({ kind: "unresolved-ref", detail: `\\ref{${label}}` });
        return "";
      }
      const prefix = label.includes(":") ? label.split(":")[0] : "";
      const typeName = labelPrefixMap[prefix] || "";
      const num = this.numberMap[label];
      const display = num
        ? `${typeName} ${num}`.trim()
        : typeName ? `${typeName}: ${title}` : title;
      return `[ref:${display}|${title}]`;
    });

    text = text.replace(/  +/g, " ");
    text = text.replace(/（\s*）/g, "");
    text = text.replace(/\(\s*\)/g, "");
    return text;
  }

  // 行内レベルの変換をまとめて適用する。
  applyInline(text, convertReferences = true) {
    // \blockmeta{...} は proofgraph 用のメタデータ。PDF に出ない以上、
    // サイトの markdown にも漏らさない。
    text = text.replace(/\\blockmeta\{[^}]*\}/g, "");
    text = stripLabel(text);
    if (convertReferences && Object.keys(this.labelMap).length > 0) {
      text = this.convertRefs(text);
    } else {
      text = stripRef(text);
    }
    text = text.replaceAll("~", " ");
    text = convertTextCommands(text);
    text = text.replace(/\\paragraph\{([^}]*)\}/g, "\n**$1**\n");
    text = convertTexorpdfstring(text);
    text = convertInlineMath(text);
    return text;
  }

  // -------------------------------------------------------------------------
  // Rendering
  // -------------------------------------------------------------------------

  // リスト項目を、内容の順序（テキスト・数式・テキスト…）を保って出力する。
  renderListItem(output, prefix, itemNodes) {
    let firstText = true;
    const textBuf = [];

    const flushText = () => {
      if (textBuf.length === 0) return;
      const text = this.applyInline(textBuf.filter((t) => t).join(""));
      if (text) {
        output.push(firstText ? `${prefix}${text}` : text);
        firstText = false;
      }
      textBuf.length = 0;
    };

    for (const n of itemNodes) {
      if (n[0] === "text") {
        textBuf.push(n[1].trim());
      } else if (n[0] === "blank") {
        continue;
      } else {
        flushText();
        for (const bl of this.renderNodes([n])) output.push(bl);
      }
    }

    flushText();
    if (firstText) output.push(prefix.replace(/\s+$/, ""));
  }

  renderNodes(nodes) {
    const { blockEnvs, envToPrefix } = this.config;
    const output = [];
    let textBuf = [];

    const flushText = () => {
      // 段落内で連続するテキスト行を 1 行に連結する。
      // markdown.mjs は段落をスペースで連結するため、ここで連結しておかないと
      // 行折り返し位置に日本語の余計な空白が入る。CJK 想定で空文字連結。
      if (textBuf.length > 0) {
        output.push(textBuf.join(""));
        textBuf = [];
      }
    };

    for (const node of nodes) {
      const kind = node[0];
      if (kind !== "text") flushText();

      if (kind === "blank") {
        output.push("");
        continue;
      }

      if (kind === "section") {
        output.push(`## ${this.applyInline(node[1])}`);
        output.push("");
        continue;
      }

      if (kind === "subsection") {
        output.push(`### ${this.applyInline(node[1])}`);
        output.push("");
        continue;
      }

      if (kind === "subsubsection") {
        output.push(`**${this.applyInline(node[1])}**`);
        output.push("");
        continue;
      }

      if (kind === "demo") {
        output.push(`:::demo ${node[1]}`);
        output.push("");
        continue;
      }

      if (kind === "block") {
        const [, envName, title, bodyNodes, proofNodes, label] = node;
        const [containerClass, prefix] = blockEnvs[envName];
        output.push(`:::${containerClass}`);
        const num = label ? this.numberMap[`${envToPrefix[envName]}:${label}`] : undefined;
        const heading = prefix
          ? `### ${prefix}${num ? ` ${num}` : ""}: ${this.applyInline(title)}`
          : `### ${this.applyInline(title)}`;
        output.push(heading);
        const bodyText = this.renderNodes(bodyNodes).join("\n").trim();
        if (bodyText) {
          output.push("");
          output.push(bodyText);
        }
        if (proofNodes !== null && proofNodes !== undefined) {
          output.push("");
          output.push(":::details-embedded 証明");
          const proofText = this.renderNodes(proofNodes).join("\n").trim();
          if (proofText) output.push(proofText);
          output.push(":::");
        }
        output.push(":::");
        output.push("");
        continue;
      }

      if (kind === "memo") {
        output.push(":::fact");
        const bodyText = this.renderNodes(node[1]).join("\n").trim();
        if (bodyText) output.push(bodyText);
        output.push(":::");
        output.push("");
        continue;
      }

      if (kind === "standalone_proof") {
        output.push(":::details-embedded 証明");
        const proofText = this.renderNodes(node[1]).join("\n").trim();
        if (proofText) output.push(proofText);
        output.push(":::");
        output.push("");
        continue;
      }

      if (kind === "display_math") {
        output.push("");
        output.push("\\[");
        for (const ml of node[1]) output.push(this.applyInline(ml, false));
        output.push("\\]");
        output.push("");
        continue;
      }

      if (kind === "align") {
        output.push("");
        output.push("\\[\\begin{aligned}");
        for (const ml of node[1]) output.push(this.applyInline(ml, false));
        output.push("\\end{aligned}\\]");
        output.push("");
        continue;
      }

      if (kind === "enumerate") {
        output.push("");
        let idx = 1;
        for (const [label, itemNodes] of node[1]) {
          this.renderListItem(output, label ? `${label} ` : `${idx}. `, itemNodes);
          // 各項目を空行で区切る。\item[(ii)] のようなラベル付き項目は
          // markdown のリスト記号にならず段落として描画されるため、区切らないと
          // 隣接する項目が 1 段落に連結され (ii) などが消えて見える。
          output.push("");
          idx += 1;
        }
        output.push("");
        continue;
      }

      if (kind === "itemize") {
        output.push("");
        for (const [label, itemNodes] of node[1]) {
          this.renderListItem(output, label ? `${label} ` : "- ", itemNodes);
          output.push("");
        }
        output.push("");
        continue;
      }

      if (kind === "text") {
        const text = node[1].trim();
        if (text) textBuf.push(this.applyInline(text));
        continue;
      }
    }

    flushText();
    return output;
  }

  // 数式の外に残った未変換マクロ（\emph 等）を検出する。
  // 数式は \(...\)（インライン）と \[ ... \]（表示ブロック）の 2 形態で、
  // その中の \command は MathJax が描画する正当なものなのでシールドする。
  lintLeftoverMacros(body, mdFilename) {
    const lines = body.split("\n");
    let inDisplay = false;
    for (let i = 0; i < lines.length; i++) {
      const trimmed = lines[i].trim();
      if (inDisplay) {
        if (trimmed.includes("\\]")) inDisplay = false;
        continue; // 表示数式ブロックの中身は生 TeX が正当
      }
      if (trimmed.startsWith("\\[")) {
        if (!trimmed.slice(2).includes("\\]")) inDisplay = true;
        continue;
      }
      const shielded = lines[i]
        .replace(/\\\([^]*?\\\)/g, " ") // インライン数式
        .replace(/\\\[[^]*?\\\]/g, " "); // 保険: 単一行の表示数式
      for (const m of shielded.matchAll(/\\[a-zA-Z]+/g)) {
        if (this.macroWhitelist.has(m[0])) continue;
        this.warn({
          kind: "leftover-macro",
          file: mdFilename,
          line: i + 1,
          detail: `${m[0]}  | ${trimmed.slice(0, 80)}`,
        });
      }
    }
  }

  // 1 章を変換して content/*.md を書き出す。ブロック数を返す。
  processChapter(chapter) {
    this.currentFile = chapter.md;
    const texPath = path.join(this.config.texDir, chapter.tex);
    const mdPath = path.join(this.config.contentDir, chapter.md);

    if (!existsSync(texPath)) {
      console.log(`  SKIP (not found): ${chapter.tex}`);
      return null;
    }

    // \qedhere は PDF 専用の証明終端記号（数式内にも現れる）。
    // MathJax は解釈できないため、変換前に除去する。
    const rawContent = readFileSync(texPath, "utf-8").replace(/\\qedhere\b/g, "");
    const rawLines = rawContent.split("\n");
    // ファイルが改行で終わる場合に末尾の空要素を作らない。
    if (rawLines.length > 0 && rawLines[rawLines.length - 1] === "") rawLines.pop();

    let lines = rawLines.map((line) => stripComments(line.replace(/\n$/, "")));
    lines = joinMultilineInlineMath(lines);

    const nodes = new TexParser(lines, this.config).parse();
    const body = this.renderNodes(nodes).join("\n");

    // 未変換マクロを検出（未解決 ref は convertRefs 内で検出済み）
    this.lintLeftoverMacros(body, chapter.md);

    const fm =
      "---\n" +
      `id: ${chapter.id}\n` +
      `group: ${chapter.group}\n` +
      `nav: ${chapter.nav}\n` +
      `eyebrow: ${chapter.eyebrow}\n` +
      `title: ${chapter.title}\n` +
      "---\n\n";

    writeFileSync(mdPath, cleanOutput(fm + body), "utf-8");
    return nodes.filter((n) => n[0] === "block").length;
  }

  // 全章を変換する。
  run({ quiet = false } = {}) {
    const log = quiet ? () => {} : (...a) => console.log(...a);

    this.labelMap = this.buildLabelMap();
    this.chapterMap = this.buildChapterMap();
    this.numberMap = this.buildNumberMap();
    log(
      `Built label map with ${Object.keys(this.labelMap).length} entries, ` +
      `chapter map with ${Object.keys(this.chapterMap).length} entries`,
    );

    mkdirSync(this.config.contentDir, { recursive: true });
    // content/*.md は全て生成物。再生成前に一掃し、旧構成の stale な md を残さない。
    let removed = 0;
    for (const name of readdirSync(this.config.contentDir)) {
      if (name.endsWith(".md")) {
        rmSync(path.join(this.config.contentDir, name));
        removed += 1;
      }
    }
    log(`Removed ${removed} stale markdown file(s) for regeneration`);
    log();

    let totalBlocks = 0;
    for (const chapter of this.config.chapters) {
      log(`Converting ${chapter.tex} -> ${chapter.md}`);
      const before = this.warnings.length;
      const count = this.processChapter(chapter);
      if (count !== null) {
        totalBlocks += count;
        const w = this.warnings.length - before;
        log(`  ${count} block(s) written` + (w ? `  ⚠ ${w} warning(s)` : "  ✓ clean"));
      }
      log();
    }

    log(
      `Done. Generated ${this.config.chapters.length} file(s), ${totalBlocks} total block(s).`,
    );
    return { totalBlocks, warnings: this.warnings };
  }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

class TexParser {
  constructor(lines, config) {
    this.lines = lines;
    this.config = config;
    this.pos = 0;
  }

  atEnd() {
    return this.pos >= this.lines.length;
  }

  peek() {
    return this.atEnd() ? "" : this.lines[this.pos];
  }

  advance() {
    const line = this.lines[this.pos];
    this.pos += 1;
    return line;
  }

  parse() {
    const nodes = [];
    this.parseBody(nodes, null);
    return nodes;
  }

  parseBody(nodes, stopEnv) {
    while (!this.atEnd()) {
      const line = this.peek();
      const stripped = line.trim();

      if (stopEnv && stripped === `\\end{${stopEnv}}`) {
        this.advance();
        return;
      }

      if (!stripped) {
        this.advance();
        while (!this.atEnd() && !this.peek().trim()) this.advance();
        nodes.push(["blank"]);
        continue;
      }

      // \demohint{NAME} → サイト側のデモブロック指示
      const mDemo = stripped.match(/^\\demohint\{([^}]+)\}/);
      if (mDemo) {
        this.advance();
        nodes.push(["demo", mDemo[1]]);
        continue;
      }

      if (stripped.startsWith("\\chapter{")) {
        this.advance();
        if (!this.atEnd() && this.peek().trim().startsWith("\\label{")) this.advance();
        continue;
      }

      let m = stripped.match(/^\\section\*?\{(.+)\}/);
      if (m) {
        this.advance();
        if (!this.atEnd() && this.peek().trim().startsWith("\\label{")) this.advance();
        nodes.push(["section", m[1]]);
        continue;
      }

      m = stripped.match(/^\\subsection\*?\{(.+)\}/);
      if (m) {
        this.advance();
        if (!this.atEnd() && this.peek().trim().startsWith("\\label{")) this.advance();
        nodes.push(["subsection", m[1]]);
        continue;
      }

      m = stripped.match(/^\\subsubsection\*?\{(.+)\}/);
      if (m) {
        this.advance();
        if (!this.atEnd() && this.peek().trim().startsWith("\\label{")) this.advance();
        nodes.push(["subsubsection", m[1]]);
        continue;
      }

      // 図は tikz のためサイトでは省略する
      if (/^\\begin\{figure\}/.test(stripped)) {
        this.skipEnvironment("figure");
        continue;
      }
      if (/^\\begin\{tikzpicture\}/.test(stripped)) {
        this.skipEnvironment("tikzpicture");
        continue;
      }
      if (stripped === "\\begin{center}") {
        this.skipEnvironment("center");
        continue;
      }

      // \begin{algorithm}{label} ... \end{algorithm}（引数はラベル）
      const mAlg = stripped.match(/^\\begin\{algorithm\}\{(.+?)\}/);
      if (mAlg) {
        this.advance();
        const raw = [];
        while (!this.atEnd() && this.peek().trim() !== "\\end{algorithm}") {
          raw.push(this.advance());
        }
        if (!this.atEnd()) this.advance();
        // 行末の '\\'（改行）を空行＝段落区切りに変換し、各ステップを
        // 別行で描画する（生の <br> は markdown.mjs に escape されるため使えない）。
        const processed = [];
        for (const ln of raw) {
          const s = ln.replace(/\s+$/, "");
          if (s.endsWith("\\\\")) {
            processed.push(s.slice(0, -2).replace(/\s+$/, ""));
            processed.push("");
          } else {
            processed.push(ln);
          }
        }
        const blockNodes = new TexParser(processed, this.config).parse();
        nodes.push(["block", "algorithm", "アルゴリズム", blockNodes, null]);
        continue;
      }

      // 定理系の環境: \begin{ENV}{タイトル}{ラベル}
      m = stripped.match(/^\\begin\{(\w+)\}\{(.+?)\}\{(.+?)\}/);
      if (m && m[1] in this.config.blockEnvs) {
        const envName = m[1];
        const title = m[2];
        const label = m[3]; // 見出しの定理番号の引き当てに使う
        this.advance();
        const blockNodes = [];
        this.parseBody(blockNodes, envName);
        const proofNodes = this.tryParseProof();
        nodes.push(["block", envName, title, blockNodes, proofNodes, label]);
        continue;
      }

      if (stripped === "\\begin{memo*}") {
        this.advance();
        const blockNodes = [];
        this.parseBody(blockNodes, "memo*");
        nodes.push(["memo", blockNodes]);
        continue;
      }

      // ブロックに続かない単独の証明。\begin{proof}[...] も受け付ける。
      if (/^\\begin\{proof\}/.test(stripped)) {
        this.advance();
        const proofNodes = [];
        this.parseBody(proofNodes, "proof");
        nodes.push(["standalone_proof", proofNodes]);
        continue;
      }

      if (stripped.startsWith("\\[")) {
        nodes.push(["display_math", this.collectDisplayMath()]);
        continue;
      }

      if (stripped.startsWith("\\begin{align*}")) {
        nodes.push(["align", this.collectEnvironment("align*")]);
        continue;
      }

      if (stripped === "\\begin{enumerate}") {
        this.advance();
        nodes.push(["enumerate", this.collectListItems("enumerate")]);
        continue;
      }

      if (stripped === "\\begin{itemize}") {
        this.advance();
        nodes.push(["itemize", this.collectListItems("itemize")]);
        continue;
      }

      if (/^\\(medskip|bigskip|smallskip|vspace\*?\{[^}]*\})\s*$/.test(stripped)) {
        this.advance();
        continue;
      }

      this.advance();
      nodes.push(["text", line]);
    }
  }

  // \end{envName} まで読み飛ばす（ネスト対応）。
  skipEnvironment(envName) {
    const esc = escapeRegExp(envName);
    const beginRe = new RegExp(`^\\\\begin\\{${esc}\\}`);
    const endRe = new RegExp(`^\\\\end\\{${esc}\\}`);
    let depth = 1;
    this.advance();
    while (!this.atEnd()) {
      const line = this.peek().trim();
      if (beginRe.test(line)) depth += 1;
      if (endRe.test(line)) {
        depth -= 1;
        if (depth === 0) {
          this.advance();
          // 後続の \caption / \label も読み飛ばす
          while (!this.atEnd()) {
            const nxt = this.peek().trim();
            if (nxt.startsWith("\\caption{") || nxt.startsWith("\\label{")) this.advance();
            else break;
          }
          return;
        }
      }
      this.advance();
    }
  }

  // 次の非空行が \begin{proof} ならそれを解析する。
  tryParseProof() {
    const savedPos = this.pos;
    while (!this.atEnd() && !this.peek().trim()) this.pos += 1;
    if (!this.atEnd() && /^\\begin\{proof\}/.test(this.peek().trim())) {
      this.advance();
      const proofNodes = [];
      this.parseBody(proofNodes, "proof");
      return proofNodes;
    }
    this.pos = savedPos;
    return null;
  }

  // \[ から \] までを集める。
  collectDisplayMath() {
    const lines = [];
    const firstLine = this.advance().trim();
    if (firstLine.includes("\\]")) {
      return [firstLine.replaceAll("\\[", "").replaceAll("\\]", "").trim()];
    }
    const contentAfter = firstLine.replaceAll("\\[", "").trim();
    if (contentAfter) lines.push(contentAfter);
    while (!this.atEnd()) {
      const line = this.advance();
      if (line.includes("\\]")) {
        const contentBefore = line.replaceAll("\\]", "").trim();
        if (contentBefore) lines.push(contentBefore);
        break;
      }
      lines.push(line.replace(/\s+$/, ""));
    }
    return lines;
  }

  // \begin{envName}...\end{envName} の中身を集める。
  collectEnvironment(envName) {
    const lines = [];
    const firstLine = this.advance().trim();
    const esc = escapeRegExp(envName);
    const after = firstLine.replace(new RegExp(`\\\\begin\\{${esc}\\}`), "").trim();
    if (after) lines.push(after);
    const endTag = `\\end{${envName}}`;
    while (!this.atEnd()) {
      const line = this.advance();
      if (line.includes(endTag)) {
        const before = line.replaceAll(endTag, "").trim();
        if (before) lines.push(before);
        break;
      }
      lines.push(line.replace(/\s+$/, ""));
    }
    return lines;
  }

  // \end{envName} までの項目を集める。[ラベル, ノード列] の配列を返す。
  collectListItems(envName) {
    const items = [];
    let currentLabel = null;
    let currentNodes = null;

    const flushItem = () => {
      if (currentNodes !== null) items.push([currentLabel, currentNodes]);
      currentLabel = null;
      currentNodes = null;
    };

    while (!this.atEnd()) {
      const stripped = this.peek().trim();
      if (stripped === `\\end{${envName}}`) {
        this.advance();
        flushItem();
        break;
      }

      const m = stripped.match(/^\\item(?:\[([^\]]*)\])?\s*([\s\S]*)/);
      if (m && stripped.startsWith("\\item")) {
        flushItem();
        currentLabel = m[1] !== undefined ? m[1] : null;
        const rest = m[2].trim();
        currentNodes = [];
        if (rest) currentNodes.push(["text", rest]);
        this.advance();
      } else {
        if (currentNodes === null) currentNodes = [];
        this.parseItemContent(currentNodes);
      }
    }

    return items;
  }

  parseItemContent(nodes) {
    const stripped = this.peek().trim();

    if (!stripped) {
      this.advance();
      nodes.push(["blank"]);
      return;
    }
    if (stripped.startsWith("\\[")) {
      nodes.push(["display_math", this.collectDisplayMath()]);
      return;
    }
    if (stripped === "\\begin{align*}") {
      nodes.push(["align", this.collectEnvironment("align*")]);
      return;
    }
    if (stripped === "\\begin{enumerate}") {
      this.advance();
      nodes.push(["enumerate", this.collectListItems("enumerate")]);
      return;
    }
    if (stripped === "\\begin{itemize}") {
      this.advance();
      nodes.push(["itemize", this.collectListItems("itemize")]);
      return;
    }
    // \begin{cases|pmatrix|bmatrix} などはテキストとして通す
    nodes.push(["text", this.advance()]);
  }
}

// ---------------------------------------------------------------------------
// Post-processing
// ---------------------------------------------------------------------------

// 空行を 2 行までに畳み、末尾を整える。
function cleanOutput(text) {
  const lines = text.split("\n").map((line) => line.replace(/\s+$/, ""));
  const cleaned = [];
  let blankCount = 0;
  for (const line of lines) {
    if (line === "") {
      blankCount += 1;
      if (blankCount <= 2) cleaned.push(line);
    } else {
      blankCount = 0;
      cleaned.push(line);
    }
  }
  return cleaned.join("\n").trim() + "\n";
}

// エスケープされていない $ の個数を数える。
function countUnescapedDollars(line) {
  let count = 0;
  for (let i = 0; i < line.length; i++) {
    if (line[i] === "$" && (i === 0 || line[i - 1] !== "\\")) count += 1;
  }
  return count;
}

// $ のインライン数式が複数行にまたがる場合に行を連結する。
function joinMultilineInlineMath(lines) {
  const result = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const stripped = line.trim();
    if (stripped.startsWith("\\[") || stripped.startsWith("\\begin{")) {
      result.push(line);
      i += 1;
      continue;
    }
    let nDollars = countUnescapedDollars(line);
    if (nDollars % 2 === 1) {
      let joined = line;
      i += 1;
      while (i < lines.length) {
        joined = joined.replace(/\s+$/, "") + " " + lines[i].trim();
        nDollars += countUnescapedDollars(lines[i]);
        i += 1;
        if (nDollars % 2 === 0) break;
      }
      result.push(joined);
    } else {
      result.push(line);
      i += 1;
    }
  }
  return result;
}

export { Converter };
