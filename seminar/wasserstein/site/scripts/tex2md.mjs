// Convert TeX seminar chapter files into site content markdown files.
//
// tex2md.py の JS 移植。出力する content/*.md の方言（frontmatter + 独自ブロック
// 記法）は build.mjs がそのままパースする中間生成物。tex が source of truth。

import { readFileSync, writeFileSync, readdirSync, existsSync, rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..", "..");
const SEMINAR_DIR = path.join(REPO_ROOT, "seminar", "wasserstein", "tex");
const CONTENT_DIR = path.join(REPO_ROOT, "seminar", "wasserstein", "site", "content");

// 本編(main/) と 付録(foundations/) の全 tex を出現順に返す。
// ラベル写像・章写像の走査対象（preamble.tex / main.tex は含めない）。
function allChapterTex() {
  const paths = [];
  for (const sub of ["main", "foundations"]) {
    const dir = path.join(SEMINAR_DIR, sub);
    if (!existsSync(dir)) continue;
    for (const name of readdirSync(dir)) {
      if (name.endsWith(".tex")) paths.push(path.join(dir, name));
    }
  }
  paths.sort();
  return paths;
}

// 変換対象の章。本編(main/) と 付録:前提知識(foundations/) を tex から生成する
// （content/*.md は tex の生成物であり、tex が source of truth）。
// group=main は本編、group=appendix は付録ナビに振り分けられる（build.mjs）。
// 参照（\ref）解決は buildLabelMap()／buildChapterMap() が main/ と foundations/ の
// 全 tex を走査するため、付録のラベルも本編からタイトル表示・参照できる。
const CHAPTERS = [
  ["main/00_introduction.tex", "00-introduction.md", {
    id: "introduction", group: "main",
    nav: "導入", eyebrow: "0. Introduction",
    title: "導入",
  }],
  ["main/01_wasserstein_metrics.tex", "01-wasserstein-metrics.md", {
    id: "wasserstein-metrics", group: "main",
    nav: "Wₚ の距離性", eyebrow: "1. The Metric Wₚ",
    title: "Wasserstein 距離 Wₚ",
  }],
  ["main/03_concentration.tex", "02-concentration.md", {
    id: "concentration", group: "main",
    nav: "Gaussian と集中", eyebrow: "2. Concentration",
    title: "Talagrand の不等式と Gaussian 測度の集中",
  }],
  ["main/04_latent.tex", "03-latent.md", {
    id: "latent", group: "main",
    nav: "潜在空間への応用", eyebrow: "3. Latent OT",
    title: "応用：潜在空間の最適輸送幾何",
  }],
  ["foundations/00_preliminaries.tex", "A0-preliminaries.md", {
    id: "found-preliminaries", group: "appendix",
    nav: "距離空間と測度", eyebrow: "付録 A. Metric Spaces & Measures",
    title: "距離空間と測度の準備",
  }],
];

// Named block environments and their markdown mappings.
// env_name -> [container_class, heading_prefix]
const BLOCK_ENVS = {
  definition: ["definition", "Def"],
  claim: ["theorem", "Clm"],
  lemma: ["theorem", "Lem"],
  theorem: ["theorem", "Thm"],
  proposition: ["theorem", "Prop"],
  corollary: ["theorem", "Cor"],
  remark: ["fact", "Rem"],
  example: ["fact accent", "Ex"],
  algorithm: ["definition", ""],
};

const LABEL_PREFIX_MAP = {
  def: "Def",
  clm: "Clm",
  lem: "Lem",
  thm: "Thm",
  prop: "Prop",
  cor: "Cor",
  rem: "Rem",
  ex: "Ex",
};

const JP_TO_ABBREV = {
  "定義": "Def", "主張": "Clm", "命題": "Prop",
  "定理": "Thm", "例": "Ex", "注意": "Rem", "補題": "Lem", "Claim": "Clm",
  "系": "Cor",
};

const ENV_TO_PREFIX = {
  definition: "def",
  claim: "clm",
  lemma: "lem",
  theorem: "thm",
  proposition: "prop",
  corollary: "cor",
  remark: "rem",
  example: "ex",
};

let LABEL_MAP = {};
let CHAPTER_MAP = {};
let NUMBER_MAP = {};

// ---------------------------------------------------------------------------
// Lint: 未変換マクロ・未解決参照を表面化させる（無警告の素通りを防ぐ）
// ---------------------------------------------------------------------------
const WARNINGS = [];
let CURRENT_FILE = null;
// 「数式の外に出ても安全」と確認できたマクロだけここに足す。空のままにすること
// （埋めると \emph 級の素通りバグを再び隠してしまう）。
const MACRO_WHITELIST = new Set([]);
function pushWarning(w) {
  WARNINGS.push({ file: CURRENT_FILE, ...w });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Remove TeX line comments (% ...), preserving escaped \%.
function stripComments(line) {
  const result = [];
  for (let i = 0; i < line.length; i++) {
    if (line[i] === "%" && (i === 0 || line[i - 1] !== "\\")) break;
    result.push(line[i]);
  }
  return result.join("").replace(/\s+$/, "");
}

// Convert $...$ to \(...\), but leave $$...$$ alone (shouldn't appear).
function convertInlineMath(text) {
  const parts = [];
  let inMath = false;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "$" && (i === 0 || text[i - 1] !== "\\")) {
      if (inMath) {
        parts.push("\\)");
        inMath = false;
      } else {
        parts.push("\\(");
        inMath = true;
      }
    } else {
      parts.push(text[i]);
    }
  }
  // If we ended with an unclosed math, revert (multi-line $ shouldn't happen
  // after line-joining, but be safe)
  if (inMath) return text;
  return parts.join("");
}

// Convert \textbf{X} -> **X**, \textit{X}/\emph{X} -> *X*, \textrm{X} -> X.
// Handles one level of nested braces (e.g. \textbf{...\mathbf{P}...}).
function convertTextCommands(text) {
  const nested = "(?:[^{}]|\\{[^{}]*\\})*";
  text = text.replace(new RegExp(`\\\\textbf\\{(${nested})\\}`, "g"), "**$1**");
  text = text.replace(new RegExp(`\\\\textit\\{(${nested})\\}`, "g"), "*$1*");
  text = text.replace(new RegExp(`\\\\emph\\{(${nested})\\}`, "g"), "*$1*");
  text = text.replace(new RegExp(`\\\\textrm\\{(${nested})\\}`, "g"), "$1");
  return text;
}

// Remove \label{...} commands.
function stripLabel(text) {
  return text.replace(/\\label\{[^}]*\}/g, "");
}

// Remove \ref{...} commands and clean up surrounding artifacts.
function stripRef(text) {
  text = text.replace(/第~?\\ref\{[^}]*\}~?章/g, "");
  text = text.replace(/§~?\\ref\{[^}]*\}/g, "");
  text = text.replace(/~?\\ref\{[^}]*\}/g, "");
  text = text.replace(/  +/g, " ");
  text = text.replace(/（\s*）/g, "");
  text = text.replace(/\(\s*\)/g, "");
  return text;
}

// Extract a brace-balanced {…} argument starting at *start*.
// Returns [content, endIndex] or null if *start* is not '{'.
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

// Scan every TeX chapter file and build a map: 'prefix:label' -> title.
function buildLabelMap() {
  const labelMap = {};
  for (const texPath of allChapterTex()) {
    if (!existsSync(texPath)) continue;
    const content = readFileSync(texPath, "utf-8");
    for (const m of content.matchAll(/\\begin\{(\w+)\}/g)) {
      const envName = m[1];
      if (!(envName in ENV_TO_PREFIX)) continue;
      let pos = m.index + m[0].length;
      const titleResult = extractBraceArg(content, pos);
      if (titleResult === null) continue;
      const title = titleResult[0];
      pos = titleResult[1];
      const labelResult = extractBraceArg(content, pos);
      if (labelResult === null) continue;
      const label = labelResult[0];
      const prefix = ENV_TO_PREFIX[envName];
      labelMap[`${prefix}:${label}`] = title;
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
function buildNumberMap() {
  const map = {};
  let mainNo = 0;
  let appendixNo = 0;
  for (const [texRel, , meta] of CHAPTERS) {
    const texPath = path.join(SEMINAR_DIR, texRel);
    if (!existsSync(texPath)) continue;
    let chapterNo;
    if (meta.group === "appendix") {
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
      if (!(envName in ENV_TO_PREFIX)) continue;
      const pos = m.index + m[0].length;
      const titleResult = extractBraceArg(content, pos);
      if (titleResult === null) continue;
      const labelResult = extractBraceArg(content, titleResult[1]);
      if (labelResult === null) continue;
      counter += 1;
      map[`${ENV_TO_PREFIX[envName]}:${labelResult[0]}`] =
        `${chapterNo}.${sectionNo}.${counter}`;
    }
  }
  return map;
}

// \texorpdfstring{A}{B} -> A など、章タイトル中の表示用整形を除く。
function cleanChapterTitle(title) {
  const nested = "(?:[^{}]|\\{[^{}]*\\})*";
  title = title.replace(
    new RegExp(`\\\\texorpdfstring\\{(${nested})\\}\\{${nested}\\}`, "g"),
    "$1",
  );
  return title.trim();
}

// Scan every TeX chapter file for \chapter{TITLE}\label{ch:...} pairs.
function buildChapterMap() {
  const chapterMap = {};
  for (const texPath of allChapterTex()) {
    const content = readFileSync(texPath, "utf-8");
    for (const m of content.matchAll(/\\chapter\{/g)) {
      const res = extractBraceArg(content, m.index + m[0].length - 1);
      if (res === null) continue;
      const [title, pos] = res;
      const labelM = content.slice(pos, pos + 200).match(/^\s*\\label\{(ch:[^}]*)\}/);
      if (!labelM) continue;
      chapterMap[labelM[1]] = cleanChapterTitle(title);
    }
  }
  return chapterMap;
}

// Convert \ref{...} to clickable [ref:display|name] links.
function convertRefs(text) {
  text = text.replace(/第~?\\ref\{(ch:[^}]*)\}~?章/g, (_, label) => {
    const title = CHAPTER_MAP[label];
    return title ? `「${title}」の章` : "本章";
  });
  text = text.replace(/§~?\\ref\{sec:[^}]*\}/g, "本節");
  text = text.replace(/(?:Algorithm|アルゴリズム)~?\\ref\{alg:[^}]*\}/g, "アルゴリズム");
  // 図は tikz のためサイトでは省略される。図参照は語「図」に落として壊さない。
  text = text.replace(/図~?\\ref\{fig:[^}]*\}/g, "図");

  text = text.replace(
    /(定義|主張|命題|定理|例|注意|補題|系|Claim|正則化問題)~?\\ref\{([^}]+)\}/g,
    (_, g1, label) => {
      const title = LABEL_MAP[label];
      if (!title) {
        pushWarning({ kind: "unresolved-ref", detail: `${g1}~\\ref{${label}}` });
        return "";
      }
      const prefix = label.includes(":") ? label.split(":")[0] : "";
      // 参照語（命題/補題/…）とラベル接頭辞（prop/lem/…）の食い違いを検出
      const expected = JP_TO_ABBREV[g1];
      const actual = LABEL_PREFIX_MAP[prefix];
      if (expected && actual && expected !== actual) {
        pushWarning({
          kind: "ref-type-mismatch",
          detail: `${g1}~\\ref{${label}}: 参照語「${g1}」(${expected}) がラベル接頭辞「${prefix}」(${actual}) と不一致`,
        });
      }
      const abbrev = LABEL_PREFIX_MAP[prefix] || (JP_TO_ABBREV[g1] ?? g1);
      // PDF と同じ番号があれば「補題 2.2.3」の形で短く示す（クリックで右に本文）。
      const num = NUMBER_MAP[label];
      const display = num ? `${g1} ${num}` : `${abbrev}: ${title}`;
      return `[ref:${display}|${title}]`;
    },
  );

  text = text.replace(/~?\\ref\{([^}]+)\}/g, (_, label) => {
    const title = LABEL_MAP[label];
    if (!title) {
      pushWarning({ kind: "unresolved-ref", detail: `\\ref{${label}}` });
      return "";
    }
    const prefix = label.includes(":") ? label.split(":")[0] : "";
    const typeName = LABEL_PREFIX_MAP[prefix] || "";
    const num = NUMBER_MAP[label];
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

// Convert non-breaking space ~ to regular space.
function convertTilde(text) {
  return text.replaceAll("~", " ");
}

// Convert \paragraph{Title.} to **Title.** on its own line.
function convertParagraph(text) {
  return text.replace(/\\paragraph\{([^}]*)\}/g, "\n**$1**\n");
}

// \texorpdfstring{A}{B} -> A（見出し・本文中に現れる表示用整形を除く）。
function convertTexorpdfstring(text) {
  const nested = "(?:[^{}]|\\{[^{}]*\\})*";
  return text.replace(
    new RegExp(`\\\\texorpdfstring\\{(${nested})\\}\\{${nested}\\}`, "g"),
    "$1",
  );
}

// Apply all inline-level conversions to a line of text.
function applyInlineConversions(text, convertReferences = true) {
  // \blockmeta{...} is proofgraph metadata: invisible in the PDF, and likewise
  // must not leak into the site markdown.
  text = text.replace(/\\blockmeta\{[^}]*\}/g, "");
  text = stripLabel(text);
  if (convertReferences && Object.keys(LABEL_MAP).length > 0) {
    text = convertRefs(text);
  } else {
    text = stripRef(text);
  }
  text = convertTilde(text);
  text = convertTextCommands(text);
  text = convertParagraph(text);
  text = convertTexorpdfstring(text);
  text = convertInlineMath(text);
  return text;
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

class TexParser {
  constructor(lines) {
    this.lines = lines;
    this.pos = 0;
  }

  atEnd() {
    return this.pos >= this.lines.length;
  }

  peek() {
    if (this.atEnd()) return "";
    return this.lines[this.pos];
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

      // Check for end of enclosing environment
      if (stopEnv && stripped === `\\end{${stopEnv}}`) {
        this.advance();
        return;
      }

      // Skip blank lines (emit paragraph break)
      if (!stripped) {
        this.advance();
        while (!this.atEnd() && !this.peek().trim()) this.advance();
        nodes.push(["blank"]);
        continue;
      }

      // \demohint{NAME}  →  emit a demo block marker for the site
      const mDemo = stripped.match(/^\\demohint\{([^}]+)\}/);
      if (mDemo) {
        this.advance();
        nodes.push(["demo", mDemo[1]]);
        continue;
      }

      // Skip \chapter
      if (stripped.startsWith("\\chapter{")) {
        this.advance();
        if (!this.atEnd() && this.peek().trim().startsWith("\\label{")) {
          this.advance();
        }
        continue;
      }

      // Section headings
      let m = stripped.match(/^\\section\*?\{(.+)\}/);
      if (m) {
        this.advance();
        if (!this.atEnd() && this.peek().trim().startsWith("\\label{")) {
          this.advance();
        }
        nodes.push(["section", m[1]]);
        continue;
      }

      m = stripped.match(/^\\subsection\*?\{(.+)\}/);
      if (m) {
        this.advance();
        if (!this.atEnd() && this.peek().trim().startsWith("\\label{")) {
          this.advance();
        }
        nodes.push(["subsection", m[1]]);
        continue;
      }

      m = stripped.match(/^\\subsubsection\*?\{(.+)\}/);
      if (m) {
        this.advance();
        if (!this.atEnd() && this.peek().trim().startsWith("\\label{")) {
          this.advance();
        }
        nodes.push(["subsubsection", m[1]]);
        continue;
      }

      // Skip figure environments (including those with optional args)
      if (/^\\begin\{figure\}/.test(stripped)) {
        this.skipEnvironment("figure");
        continue;
      }

      // Skip standalone tikzpicture (not inside figure)
      if (/^\\begin\{tikzpicture\}/.test(stripped)) {
        this.skipEnvironment("tikzpicture");
        continue;
      }

      // Skip standalone center that contains tikzpicture or tabular
      if (stripped === "\\begin{center}") {
        this.skipEnvironment("center");
        continue;
      }

      // Algorithm environments -> rendered as a block.
      // \begin{algorithm}{label} ... \end{algorithm}（引数はラベル）。
      const mAlg = stripped.match(/^\\begin\{algorithm\}\{(.+?)\}/);
      if (mAlg) {
        this.advance();
        const raw = [];
        while (!this.atEnd() && this.peek().trim() !== "\\end{algorithm}") {
          raw.push(this.advance());
        }
        if (!this.atEnd()) this.advance(); // consume \end{algorithm}
        // 行末の '\\'（改行）を空行＝段落区切りに変換し、各ステップを
        // 別行で描画する（生の <br> は build.mjs に escape されるため使えない）。
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
        const blockNodes = new TexParser(processed).parse();
        nodes.push(["block", "algorithm", "アルゴリズム", blockNodes, null]);
        continue;
      }

      // Named block environments: definition, claim, theorem, etc.
      m = stripped.match(/^\\begin\{(\w+)\}\{(.+?)\}\{(.+?)\}/);
      if (m && m[1] in BLOCK_ENVS) {
        const envName = m[1];
        const title = m[2];
        const label = m[3]; // 見出しの定理番号（NUMBER_MAP）の引き当てに使う
        this.advance();
        const blockNodes = [];
        this.parseBody(blockNodes, envName);
        // Check if next thing is a proof that belongs to this block
        const proofNodes = this.tryParseProof();
        nodes.push(["block", envName, title, blockNodes, proofNodes, label]);
        continue;
      }

      // memo* environment -> fact block
      if (stripped === "\\begin{memo*}") {
        this.advance();
        const blockNodes = [];
        this.parseBody(blockNodes, "memo*");
        nodes.push(["memo", blockNodes]);
        continue;
      }

      // Standalone proof (not following a block -- rare but possible).
      // Optional argument \begin{proof}[...] も受け付ける。
      if (/^\\begin\{proof\}/.test(stripped)) {
        this.advance();
        const proofNodes = [];
        this.parseBody(proofNodes, "proof");
        nodes.push(["standalone_proof", proofNodes]);
        continue;
      }

      // Display math: \[ ... \]
      if (stripped.startsWith("\\[")) {
        const mathLines = this.collectDisplayMath();
        nodes.push(["display_math", mathLines]);
        continue;
      }

      // align* environment
      if (stripped.startsWith("\\begin{align*}")) {
        const mathLines = this.collectEnvironment("align*");
        nodes.push(["align", mathLines]);
        continue;
      }

      // enumerate
      if (stripped === "\\begin{enumerate}") {
        this.advance();
        const items = this.collectListItems("enumerate");
        nodes.push(["enumerate", items]);
        continue;
      }

      // itemize
      if (stripped === "\\begin{itemize}") {
        this.advance();
        const items = this.collectListItems("itemize");
        nodes.push(["itemize", items]);
        continue;
      }

      // Skip vertical spacing commands
      if (/^\\(medskip|bigskip|smallskip|vspace\*?\{[^}]*\})\s*$/.test(stripped)) {
        this.advance();
        continue;
      }

      // Regular text line
      this.advance();
      nodes.push(["text", line]);
    }
  }

  // Skip past \end{envName}, handling nesting.
  skipEnvironment(envName) {
    const esc = escapeRegExp(envName);
    const beginRe = new RegExp(`^\\\\begin\\{${esc}\\}`);
    const endRe = new RegExp(`^\\\\end\\{${esc}\\}`);
    let depth = 1;
    this.advance(); // consume \begin line
    while (!this.atEnd()) {
      const line = this.peek().trim();
      if (beginRe.test(line)) depth += 1;
      if (line === `\\end{${envName}}` || endRe.test(line)) {
        depth -= 1;
        if (depth === 0) {
          this.advance();
          // Also skip \caption and \label lines that follow
          while (!this.atEnd()) {
            const nxt = this.peek().trim();
            if (nxt.startsWith("\\caption{") || nxt.startsWith("\\label{")) {
              this.advance();
            } else {
              break;
            }
          }
          return;
        }
      }
      this.advance();
    }
  }

  // If the next non-blank content is \begin{proof}, parse it.
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

  // Collect lines of display math from \[ to \].
  collectDisplayMath() {
    const lines = [];
    const firstLine = this.advance().trim();
    if (firstLine.includes("\\]")) {
      const content = firstLine.replaceAll("\\[", "").replaceAll("\\]", "").trim();
      return [content];
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

  // Collect all lines inside \begin{envName}...\end{envName}.
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

  // Collect list items until \end{envName}. Returns list of [label, nodeList].
  collectListItems(envName) {
    const items = [];
    let currentLabel = null;
    let currentNodes = null;

    const flushItem = () => {
      if (currentNodes !== null) {
        items.push([currentLabel, currentNodes]);
      }
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
        this.parseItemContent(currentNodes, envName);
      }
    }

    return items;
  }

  // Parse a single piece of content within a list item.
  parseItemContent(nodes, listEnvName) {
    const stripped = this.peek().trim();

    if (!stripped) {
      this.advance();
      nodes.push(["blank"]);
      return;
    }

    if (stripped.startsWith("\\[")) {
      const mathLines = this.collectDisplayMath();
      nodes.push(["display_math", mathLines]);
      return;
    }

    if (stripped === "\\begin{align*}") {
      const mathLines = this.collectEnvironment("align*");
      nodes.push(["align", mathLines]);
      return;
    }

    if (stripped === "\\begin{enumerate}") {
      this.advance();
      const items = this.collectListItems("enumerate");
      nodes.push(["enumerate", items]);
      return;
    }

    if (stripped === "\\begin{itemize}") {
      this.advance();
      const items = this.collectListItems("itemize");
      nodes.push(["itemize", items]);
      return;
    }

    // \begin{cases|pmatrix|bmatrix} falls through to text

    nodes.push(["text", this.advance()]);
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

// Render a list item preserving content order (text, math, text, ...).
function renderListItem(output, prefix, itemNodes) {
  let firstText = true;
  const textBuf = [];

  const flushText = () => {
    if (textBuf.length === 0) return;
    let text = textBuf.filter((t) => t).join("");
    text = applyInlineConversions(text);
    if (text) {
      if (firstText) {
        output.push(`${prefix}${text}`);
        firstText = false;
      } else {
        output.push(text);
      }
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
      const blockLines = renderNodes([n]);
      for (const bl of blockLines) output.push(bl);
    }
  }

  flushText();
  if (firstText) output.push(prefix.replace(/\s+$/, ""));
}

// Render parsed nodes into markdown lines.
function renderNodes(nodes) {
  const output = [];
  let textBuf = [];

  const flushText = () => {
    // 段落内で連続するテキスト行を 1 行に連結する。
    // build.mjs は段落をスペースで連結するため、ここで連結しておかないと
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
      const title = applyInlineConversions(node[1]);
      output.push(`## ${title}`);
      output.push("");
      continue;
    }

    if (kind === "subsection") {
      const title = applyInlineConversions(node[1]);
      output.push(`### ${title}`);
      output.push("");
      continue;
    }

    if (kind === "subsubsection") {
      const title = applyInlineConversions(node[1]);
      output.push(`**${title}**`);
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
      const [containerClass, prefix] = BLOCK_ENVS[envName];
      output.push(`:::${containerClass}`);
      const num = label ? NUMBER_MAP[`${ENV_TO_PREFIX[envName]}:${label}`] : undefined;
      let heading;
      if (prefix) {
        heading = `### ${prefix}${num ? ` ${num}` : ""}: ${applyInlineConversions(title)}`;
      } else {
        heading = `### ${applyInlineConversions(title)}`;
      }
      output.push(heading);
      const bodyLines = renderNodes(bodyNodes);
      const bodyText = bodyLines.join("\n").trim();
      if (bodyText) {
        output.push("");
        output.push(bodyText);
      }
      if (proofNodes !== null && proofNodes !== undefined) {
        output.push("");
        output.push(":::details-embedded 証明");
        const proofLines = renderNodes(proofNodes);
        const proofText = proofLines.join("\n").trim();
        if (proofText) output.push(proofText);
        output.push(":::");
      }
      output.push(":::");
      output.push("");
      continue;
    }

    if (kind === "memo") {
      const bodyNodes = node[1];
      output.push(":::fact");
      const bodyLines = renderNodes(bodyNodes);
      const bodyText = bodyLines.join("\n").trim();
      if (bodyText) output.push(bodyText);
      output.push(":::");
      output.push("");
      continue;
    }

    if (kind === "standalone_proof") {
      const proofNodes = node[1];
      output.push(":::details-embedded 証明");
      const proofLines = renderNodes(proofNodes);
      const proofText = proofLines.join("\n").trim();
      if (proofText) output.push(proofText);
      output.push(":::");
      output.push("");
      continue;
    }

    if (kind === "display_math") {
      const mathLines = node[1];
      output.push("");
      output.push("\\[");
      for (const ml of mathLines) {
        output.push(applyInlineConversions(ml, false));
      }
      output.push("\\]");
      output.push("");
      continue;
    }

    if (kind === "align") {
      const mathLines = node[1];
      output.push("");
      output.push("\\[\\begin{aligned}");
      for (const ml of mathLines) {
        output.push(applyInlineConversions(ml, false));
      }
      output.push("\\end{aligned}\\]");
      output.push("");
      continue;
    }

    if (kind === "enumerate") {
      const items = node[1];
      output.push("");
      let idx = 1;
      for (const [label, itemNodes] of items) {
        const prefix = label ? `${label} ` : `${idx}. `;
        renderListItem(output, prefix, itemNodes);
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
      const items = node[1];
      output.push("");
      for (const [label, itemNodes] of items) {
        const prefix = label ? `${label} ` : "- ";
        renderListItem(output, prefix, itemNodes);
        output.push("");
      }
      output.push("");
      continue;
    }

    if (kind === "text") {
      const text = node[1].trim();
      if (text) textBuf.push(applyInlineConversions(text));
      continue;
    }
  }

  flushText();
  return output;
}

// Post-process the rendered markdown: collapse blank lines, trim.
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
  const result = cleaned.join("\n").trim();
  return result + "\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

// Count unescaped $ signs in a line (not inside \[ \] blocks).
function countUnescapedDollars(line) {
  let count = 0;
  for (let i = 0; i < line.length; i++) {
    if (line[i] === "$" && (i === 0 || line[i - 1] !== "\\")) count += 1;
  }
  return count;
}

// Join lines where $ inline math spans multiple lines.
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

// Lint: 数式の外に残った未変換マクロ（\emph 等）を検出する。
// 数式は build.mjs と同じく \(...\)（インライン）と \[ ... \]（表示ブロック）の
// 2 形態で、その中の \command は MathJax が描画する正当なものなのでシールドする。
// 注意: 表示数式 \[ は現状つねに行頭・単独行で現れる前提（renderNodes）。
//       将来それが崩れた場合に備え、単一行 \[...\] も保険でシールドしている。
function lintLeftoverMacros(body, mdFilename) {
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
      .replace(/\\\([^]*?\\\)/g, " ") // インライン数式（build.mjs:86 と同形）
      .replace(/\\\[[^]*?\\\]/g, " "); // 保険: 単一行の表示数式
    for (const m of shielded.matchAll(/\\[a-zA-Z]+/g)) {
      if (MACRO_WHITELIST.has(m[0])) continue;
      pushWarning({
        kind: "leftover-macro",
        file: mdFilename,
        line: i + 1,
        detail: `${m[0]}  | ${trimmed.slice(0, 80)}`,
      });
    }
  }
}

// Process one chapter: read TeX, parse, render, write markdown.
function processChapter(texFilename, mdFilename, frontmatter) {
  CURRENT_FILE = mdFilename;
  const texPath = path.join(SEMINAR_DIR, texFilename);
  const mdPath = path.join(CONTENT_DIR, mdFilename);

  if (!existsSync(texPath)) {
    console.log(`  SKIP (not found): ${texFilename}`);
    return null;
  }

  // \qedhere は PDF 専用の証明終端記号（数式内にも現れる）。
  // KaTeX は解釈できないため、変換前に除去する。
  const rawContent = readFileSync(texPath, "utf-8").replace(
    /\\qedhere\b/g,
    "",
  );
  let rawLines = rawContent.split("\n");
  // readlines() は末尾の空要素を生まない。ファイルが改行で終わる場合に合わせる。
  if (rawLines.length > 0 && rawLines[rawLines.length - 1] === "") {
    rawLines.pop();
  }

  // Strip comments from each line
  let lines = rawLines.map((line) => stripComments(line.replace(/\n$/, "")));

  // Join multi-line inline math
  lines = joinMultilineInlineMath(lines);

  // Parse
  const parser = new TexParser(lines);
  const nodes = parser.parse();

  // Render
  const mdLines = renderNodes(nodes);
  const body = mdLines.join("\n");

  // Lint: 未変換マクロを検出（未解決 ref は convertRefs 内で検出済み）
  lintLeftoverMacros(body, mdFilename);

  // Build frontmatter
  let fm = "---\n";
  fm += `id: ${frontmatter.id}\n`;
  fm += `group: ${frontmatter.group ?? "main"}\n`;
  fm += `nav: ${frontmatter.nav}\n`;
  fm += `eyebrow: ${frontmatter.eyebrow}\n`;
  fm += `title: ${frontmatter.title}\n`;
  fm += "---\n\n";

  let full = fm + body;
  full = cleanOutput(full);

  writeFileSync(mdPath, full, "utf-8");

  const blockCount = nodes.filter((n) => n[0] === "block").length;
  return blockCount;
}

function main() {
  LABEL_MAP = buildLabelMap();
  CHAPTER_MAP = buildChapterMap();
  NUMBER_MAP = buildNumberMap();
  if (process.env.TEX2MD_DUMP_NUMBERS) {
    console.log(JSON.stringify(NUMBER_MAP, null, 2));
  }
  console.log(
    `Built label map with ${Object.keys(LABEL_MAP).length} entries, ` +
    `chapter map with ${Object.keys(CHAPTER_MAP).length} entries`,
  );

  // content/*.md は全て生成物。再生成前に一掃し、旧構成の stale な md を残さない。
  let removed = 0;
  for (const name of readdirSync(CONTENT_DIR)) {
    if (name.endsWith(".md")) {
      rmSync(path.join(CONTENT_DIR, name));
      removed += 1;
    }
  }
  console.log(`Removed ${removed} stale markdown file(s) for regeneration`);
  console.log();

  let totalBlocks = 0;
  for (const [texFile, mdFile, fm] of CHAPTERS) {
    console.log(`Converting ${texFile} -> ${mdFile}`);
    const before = WARNINGS.length;
    const count = processChapter(texFile, mdFile, fm);
    if (count !== null) {
      totalBlocks += count;
      const w = WARNINGS.length - before;
      console.log(`  ${count} block(s) written` + (w ? `  ⚠ ${w} warning(s)` : "  ✓ clean"));
    }
    console.log();
  }

  console.log(`Done. Generated ${CHAPTERS.length} file(s), ${totalBlocks} total block(s).`);
  const generated = readdirSync(CONTENT_DIR)
    .filter((n) => n.endsWith(".md"))
    .sort();
  for (const f of generated) {
    console.log(`  ${f}`);
  }

  // ---- Lint summary ----
  if (WARNINGS.length) {
    console.warn(`\n${WARNINGS.length} lint warning(s):`);
    for (const w of WARNINGS) {
      const loc = w.line ? `${w.file}:${w.line}` : w.file;
      console.warn(`  [${w.kind}] ${loc}  ${w.detail ?? ""}`.trimEnd());
    }
    const strict =
      process.argv.includes("--strict") || process.env.TEX2MD_STRICT === "1";
    if (strict) {
      console.error(`\nFAILED: ${WARNINGS.length} lint warning(s) under --strict.`);
      process.exit(1);
    }
  } else {
    console.log("\nLint: no warnings.");
  }
}

main();
