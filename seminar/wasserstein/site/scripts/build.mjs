import { readdirSync, readFileSync, writeFileSync, mkdirSync, copyFileSync, rmSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const siteRoot = path.resolve(__dirname, "..");
const contentDir = path.join(siteRoot, "content");
const distDir = path.join(siteRoot, "dist");

// 各ページの dist 相対出力パス（本編→ main/、付録→ appendix/）。
function outPathOf(section) {
  const sub = section.data.group === "appendix" ? "appendix" : "main";
  return `${sub}/${section.data.id}.html`;
}

// from ページ（dist 相対）から to リソース（dist 相対）への相対 URL を返す。
function relUrl(fromOutPath, toOutPath) {
  const rel = path.posix.relative(path.posix.dirname(fromOutPath), toOutPath);
  return rel || ".";
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

const allBlocks = [];
const usedBlockIds = new Set();
let currentChapterId = null;

function makeBlockId(type, rawName) {
  const prefixMap = { definition: "def", theorem: "thm", proposition: "prop", lemma: "lem", remark: "rem", example: "ex" };
  const prefix = prefixMap[type] || type;
  const slug = rawName
    .replace(/\\\(([^)]*)\\\)/g, (_, tex) =>
      tex.replace(/\\[a-zA-Z]+/g, "").replace(/[{}^_]/g, "").trim()
    )
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9぀-ゟ゠-ヿ一-鿿-]/g, "")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
  let id = `${prefix}-${slug || "unnamed"}`;
  if (usedBlockIds.has(id)) {
    let n = 2;
    while (usedBlockIds.has(`${id}-${n}`)) n += 1;
    id = `${id}-${n}`;
  }
  usedBlockIds.add(id);
  return id;
}

function parseFrontmatter(source, filePath) {
  const src = source.replaceAll("\r\n", "\n");
  if (!src.startsWith("---\n")) {
    throw new Error(`${filePath}: frontmatter is required`);
  }

  const end = src.indexOf("\n---\n", 4);
  if (end === -1) {
    throw new Error(`${filePath}: frontmatter is not closed`);
  }

  const raw = src.slice(4, end);
  const body = src.slice(end + 5);
  const data = {};
  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    const index = line.indexOf(":");
    if (index === -1) {
      throw new Error(`${filePath}: invalid frontmatter line "${line}"`);
    }
    const key = line.slice(0, index).trim();
    const value = line.slice(index + 1).trim();
    data[key] = value;
  }
  return { data, body };
}

function renderInline(source) {
  const mathSpans = [];
  const shielded = source.replace(/\\\([^]*?\\\)/g, (m) => {
    mathSpans.push(m);
    return `\x00MATH${mathSpans.length - 1}\x00`;
  });

  let result = escapeHtml(shielded)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(
      /\[term:([^|\]]+)\|([a-z0-9-]+)\]/g,
      (_match, label, term) => `<button type="button" class="term" data-term="${term}">${label}</button>`
    )
    .replace(
      /\[ref:([^|\]]+?)(?:\|([^\]]+))?\]/g,
      (_match, first, second) => {
        const refName = second || first;
        // 旧形式「Lem: タイトル」は種別だけに短縮。新形式「補題 2.2.3」はそのまま。
        // ツールチップには参照先のタイトルを出す。
        const typeMatch = /^(Def|Clm|Thm|Prop|Rem|Ex|Lem|Cor)(\s+[0-9A-Z.]+)?:\s*(.+)$/.exec(first);
        const display = typeMatch ? `${typeMatch[1]}${typeMatch[2] ?? ""}` : first;
        return `<button type="button" class="ref" data-ref="${refName}" title="${escapeHtml(refName)}">${escapeHtml(display)}</button>`;
      }
    );

  return result.replace(/\x00MATH(\d+)\x00/g, (_, i) =>
    escapeHtml(mathSpans[parseInt(i)])
  );
}

function transportDiagramStyle() {
  return `
<style>
.tp-fig{position:relative;width:100%;max-width:460px;height:200px;margin:12px auto}
.tp-fig .node{position:absolute;width:76px;height:36px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;border:1.5px solid}
.tp-fig .node--f{background:#dbeafe;border-color:#3b82f6}
.tp-fig .node--s{background:#ffedd5;border-color:#f97316}
.tp-fig .lbl{position:absolute;font-size:13px;white-space:nowrap}
.tp-fig .edge-lbl{position:absolute;font-size:12px;background:rgba(255,255,255,.85);padding:0 3px;white-space:nowrap}
.tp-fig svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
</style>`;
}

function transportCostDiagram() {
  return `
${transportDiagramStyle()}
<figure aria-label="輸送コストのネットワーク" style="margin:1em 0">
  <div class="tp-fig">
    <svg viewBox="0 0 460 200" xmlns="http://www.w3.org/2000/svg">
      <defs><marker id="ac" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0,0 7,2.5 0,5" fill="#555"/></marker></defs>
      <line x1="152" y1="50" x2="295" y2="50" stroke="#555" stroke-width="1.5" marker-end="url(#ac)"/>
      <line x1="152" y1="62" x2="295" y2="155" stroke="#555" stroke-width="1.5" marker-end="url(#ac)"/>
      <line x1="152" y1="155" x2="295" y2="62" stroke="#555" stroke-width="1.5" marker-end="url(#ac)"/>
      <line x1="152" y1="165" x2="295" y2="165" stroke="#555" stroke-width="1.5" marker-end="url(#ac)"/>
    </svg>
    <span class="lbl" style="left:90px;top:2px;font-weight:bold">工場</span>
    <span class="lbl" style="left:310px;top:2px;font-weight:bold">スーパー</span>
    <div class="node node--f" style="left:75px;top:30px">\\(x_1\\)</div>
    <div class="node node--f" style="left:75px;top:145px">\\(x_2\\)</div>
    <div class="node node--s" style="left:296px;top:30px">\\(y_1\\)</div>
    <div class="node node--s" style="left:296px;top:145px">\\(y_2\\)</div>
    <span class="lbl" style="right:395px;top:38px">\\(a_1\\!=\\!\\tfrac{2}{3}\\)</span>
    <span class="lbl" style="right:395px;top:153px">\\(a_2\\!=\\!\\tfrac{1}{3}\\)</span>
    <span class="lbl" style="left:380px;top:38px">\\(b_1\\!=\\!\\tfrac{1}{3}\\)</span>
    <span class="lbl" style="left:380px;top:153px">\\(b_2\\!=\\!\\tfrac{2}{3}\\)</span>
    <span class="edge-lbl" style="left:192px;top:28px;color:#333">\\(C_{1,1}\\!=\\!1\\)</span>
    <span class="edge-lbl" style="left:170px;top:98px;color:#333">\\(C_{1,2}\\!=\\!2\\)</span>
    <span class="edge-lbl" style="left:230px;top:88px;color:#333">\\(C_{2,1}\\!=\\!3\\)</span>
    <span class="edge-lbl" style="left:192px;top:170px;color:#333">\\(C_{2,2}\\!=\\!1\\)</span>
  </div>
  <figcaption style="text-align:center;font-size:0.9em;color:#666;margin-top:4px">
    輸送コストのネットワーク．各辺の数値は単位量あたりの輸送コスト \\(C_{i,j}\\) を表す．
  </figcaption>
</figure>`;
}

function transportOptimalDiagram() {
  return `
<figure aria-label="最適輸送計画" style="margin:1em 0">
  <div class="tp-fig">
    <svg viewBox="0 0 460 200" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="ao" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0,0 7,2.5 0,5" fill="#3b82f6"/></marker>
        <marker id="ag" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0,0 7,2.5 0,5" fill="#bbb"/></marker>
      </defs>
      <line x1="152" y1="50" x2="295" y2="50" stroke="#3b82f6" stroke-width="3" marker-end="url(#ao)"/>
      <line x1="152" y1="62" x2="295" y2="155" stroke="#3b82f6" stroke-width="3" marker-end="url(#ao)"/>
      <line x1="152" y1="155" x2="295" y2="62" stroke="#bbb" stroke-width="1" stroke-dasharray="6,4" marker-end="url(#ag)"/>
      <line x1="152" y1="165" x2="295" y2="165" stroke="#3b82f6" stroke-width="3" marker-end="url(#ao)"/>
    </svg>
    <span class="lbl" style="left:90px;top:2px;font-weight:bold">工場</span>
    <span class="lbl" style="left:310px;top:2px;font-weight:bold">スーパー</span>
    <div class="node node--f" style="left:75px;top:30px">\\(x_1\\)</div>
    <div class="node node--f" style="left:75px;top:145px">\\(x_2\\)</div>
    <div class="node node--s" style="left:296px;top:30px">\\(y_1\\)</div>
    <div class="node node--s" style="left:296px;top:145px">\\(y_2\\)</div>
    <span class="lbl" style="right:395px;top:38px">\\(a_1\\!=\\!\\tfrac{2}{3}\\)</span>
    <span class="lbl" style="right:395px;top:153px">\\(a_2\\!=\\!\\tfrac{1}{3}\\)</span>
    <span class="lbl" style="left:380px;top:38px">\\(b_1\\!=\\!\\tfrac{1}{3}\\)</span>
    <span class="lbl" style="left:380px;top:153px">\\(b_2\\!=\\!\\tfrac{2}{3}\\)</span>
    <span class="edge-lbl" style="left:175px;top:28px;color:#1e40af">\\(P_{1,1}^\\star\\!=\\!\\tfrac{1}{3}\\) <span style="font-size:11px;color:#666">\\((C\\!=\\!1)\\)</span></span>
    <span class="edge-lbl" style="left:153px;top:98px;color:#1e40af">\\(P_{1,2}^\\star\\!=\\!\\tfrac{1}{3}\\) <span style="font-size:11px;color:#666">\\((C\\!=\\!2)\\)</span></span>
    <span class="edge-lbl" style="left:218px;top:88px;color:#999">\\(P_{2,1}^\\star\\!=\\!0\\) <span style="font-size:11px;color:#888">\\((C\\!=\\!3)\\)</span></span>
    <span class="edge-lbl" style="left:175px;top:170px;color:#1e40af">\\(P_{2,2}^\\star\\!=\\!\\tfrac{1}{3}\\) <span style="font-size:11px;color:#666">\\((C\\!=\\!1)\\)</span></span>
  </div>
  <figcaption style="text-align:center;font-size:0.9em;color:#666;margin-top:4px">
    最適輸送計画 \\(\\mathbf{P}^\\star\\)．安価な経路 \\(x_1 \\to y_1\\)（コスト 1）と
    \\(x_2 \\to y_2\\)（コスト 1）を最大限利用し，残りを \\(x_1 \\to y_2\\)（コスト 2）で補う．
    高コストの \\(x_2 \\to y_1\\)（コスト 3）は使われない．
  </figcaption>
</figure>`;
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  const stack = [];
  let paragraph = [];
  let listType = null;
  let currentBlock = null;

  const closeList = () => {
    if (!listType) return;
    html.push(`</${listType}>`);
    listType = null;
  };

  const flushParagraph = () => {
    if (paragraph.length === 0) return;
    html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    paragraph = [];
  };

  const openList = (type) => {
    if (listType === type) return;
    closeList();
    html.push(`<${type}>`);
    listType = type;
  };

  const closeContainer = () => {
    flushParagraph();
    closeList();
    const closing = stack.pop();
    if (!closing) {
      throw new Error("container close marker without an open container");
    }
    if (currentBlock && stack.length === currentBlock.depth) {
      if (currentBlock.name) {
        const contentHtml = html.slice(currentBlock.divIndex + 1).join("\n");
        allBlocks.push({
          id: currentBlock.id,
          name: currentBlock.name,
          type: currentBlock.type,
          title: currentBlock.fullTitle,
          chapter: currentChapterId,
          html: contentHtml
        });
      }
      currentBlock = null;
    }
    html.push(closing);
  };

  const openContainer = (spec) => {
    flushParagraph();
    closeList();

    if (spec === "grid two") {
      html.push('<div class="grid two">');
      stack.push("</div>");
      return;
    }
    if (spec === "compare") {
      html.push('<div class="compare">');
      stack.push("</div>");
      return;
    }
    if (spec === "column") {
      html.push("<div>");
      stack.push("</div>");
      return;
    }
    if (spec === "definition") {
      currentBlock = { type: "definition", divIndex: html.length, depth: stack.length };
      html.push('<div class="block block--def">');
      stack.push("</div>");
      return;
    }
    if (spec === "theorem") {
      currentBlock = { type: "theorem", divIndex: html.length, depth: stack.length };
      html.push('<div class="block block--thm">');
      stack.push("</div>");
      return;
    }
    if (spec === "proposition") {
      currentBlock = { type: "proposition", divIndex: html.length, depth: stack.length };
      html.push('<div class="block block--prop">');
      stack.push("</div>");
      return;
    }
    if (spec === "lemma") {
      currentBlock = { type: "lemma", divIndex: html.length, depth: stack.length };
      html.push('<div class="block block--thm">');
      stack.push("</div>");
      return;
    }
    if (spec === "algorithm") {
      currentBlock = { type: "algorithm", divIndex: html.length, depth: stack.length };
      html.push('<div class="block block--algo">');
      stack.push("</div>");
      return;
    }
    if (spec === "fact") {
      currentBlock = { type: "remark", divIndex: html.length, depth: stack.length };
      html.push('<aside class="margin-note">');
      stack.push("</aside>");
      return;
    }
    if (spec === "fact accent") {
      currentBlock = { type: "example", divIndex: html.length, depth: stack.length };
      html.push('<div class="example-band"><article class="example-band__inner">');
      stack.push("</article></div>");
      return;
    }
    if (spec.startsWith("details-embedded ")) {
      const title = spec.slice("details-embedded ".length);
      html.push(`<details class="proof"><summary>${renderInline(title)}</summary>`);
      stack.push("</details>");
      return;
    }
    if (spec.startsWith("details ")) {
      const title = spec.slice("details ".length);
      html.push(`<details class="fold"><summary>${renderInline(title)}</summary>`);
      stack.push("</details>");
      return;
    }
    if (spec === "demo transport-cost") {
      html.push(transportCostDiagram());
      return;
    }
    if (spec === "demo transport-optimal") {
      html.push(transportOptimalDiagram());
      return;
    }

    throw new Error(`unknown container "${spec}"`);
  };

  for (let i = 0; i < lines.length; i += 1) {
    const raw = lines[i];
    const trimmed = raw.trim();

    if (trimmed === "") {
      flushParagraph();
      let nextIndex = i + 1;
      while (nextIndex < lines.length && lines[nextIndex].trim() === "") {
        nextIndex += 1;
      }
      const next = nextIndex < lines.length ? lines[nextIndex].trim() : "";
      const continuesList =
        (listType === "ol" && /^\d+\.\s+/.test(next)) ||
        (listType === "ul" && /^[-*]\s+/.test(next));
      if (continuesList) continue;
      closeList();
      continue;
    }

    if (trimmed.startsWith("```")) {
      flushParagraph();
      closeList();
      const language = trimmed.slice(3).trim();
      const code = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        code.push(lines[i]);
        i += 1;
      }
      if (i >= lines.length) {
        throw new Error("unclosed code fence (```) at EOF");
      }
      if (language === "mermaid") {
        html.push(`<div class="map-wrap"><pre class="mermaid">${escapeHtml(code.join("\n"))}</pre></div>`);
      } else if (language === "rawhtml") {
        html.push(code.join("\n"));
      } else {
        html.push(`<pre class="code-block"><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      }
      continue;
    }

    if (trimmed === ":::") {
      closeContainer();
      continue;
    }

    if (trimmed.startsWith(":::")) {
      openContainer(trimmed.slice(3).trim());
      continue;
    }

    if (trimmed.startsWith("\\[")) {
      flushParagraph();
      closeList();
      const math = [trimmed];
      if (!trimmed.endsWith("\\]")) {
        i += 1;
        while (i < lines.length) {
          math.push(lines[i]);
          if (lines[i].trim().endsWith("\\]")) break;
          i += 1;
        }
      }
      html.push(`<div class="math-block">${escapeHtml(math.join("\n"))}</div>`);
      continue;
    }

    const heading = /^(#{2,4})\s+(.+)$/.exec(trimmed);
    if (heading) {
      flushParagraph();
      closeList();
      const level = heading[1].length;
      if (currentBlock && level === 3 && !currentBlock.name) {
        const rawTitle = heading[2];
        // 見出しは「Lem 2.2.3: タイトル」（番号付き）と「Lem: タイトル」の両形式を許す。
        // name（参照キー）は番号を含まないタイトル部分。
        const nameMatch = /^(?:Def|Clm|Thm|Prop|Rem|Ex|Lem|Cor)(?:\s+[0-9A-Z.]+)?:\s*(.+)$/.exec(rawTitle);
        const name = nameMatch ? nameMatch[1].trim() : rawTitle.trim();
        const id = makeBlockId(currentBlock.type, name);
        currentBlock.name = name;
        currentBlock.id = id;
        currentBlock.fullTitle = rawTitle;
        const original = html[currentBlock.divIndex];
        html[currentBlock.divIndex] = original.replace(/>/, ` id="${escapeHtml(id)}">`);
      }
      if (level === 2) {
        const slug = heading[2].trim().toLowerCase()
          .replace(/\\\([^)]*\\\)/g, "")
          .replace(/\s+/g, "-")
          .replace(/[^a-z0-9ぁ-ゟ゠-ヿ一-鿿-]/g, "")
          .replace(/-+/g, "-")
          .replace(/^-|-$/g, "");
        html.push(`<h2 id="sec-${slug}">${renderInline(heading[2])}</h2>`);
      } else if (level === 3 && currentBlock && GRAPH_TYPES.has(currentBlock.type) && currentBlock.id) {
        const graphLink = `<a class="block__graph-link" href="__GRAPH_BASE__?focus=${encodeURIComponent(currentBlock.id)}" title="依存グラフで表示"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="5" r="2.5"/><circle cx="5" cy="19" r="2.5"/><circle cx="19" cy="19" r="2.5"/><line x1="12" y1="7.5" x2="5" y2="16.5"/><line x1="12" y1="7.5" x2="19" y2="16.5"/></svg></a>`;
        html.push(`<h3>${renderInline(heading[2])}${graphLink}</h3>`);
      } else {
        html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      }
      continue;
    }

    const ordered = /^\d+\.\s+(.+)$/.exec(trimmed);
    if (ordered) {
      flushParagraph();
      openList("ol");
      html.push(`<li>${renderInline(ordered[1])}</li>`);
      continue;
    }

    const unordered = /^[-*]\s+(.+)$/.exec(trimmed);
    if (unordered) {
      flushParagraph();
      openList("ul");
      html.push(`<li>${renderInline(unordered[1])}</li>`);
      continue;
    }

    paragraph.push(trimmed);
  }

  flushParagraph();
  closeList();
  if (stack.length > 0) {
    throw new Error(`${stack.length} unclosed container(s) (:::) at EOF`);
  }

  return html.join("\n");
}

/* ==========================================================================
   Templates — multi-page output
   ========================================================================== */

function mathJaxScript() {
  return `<script>
      window.MathJax = {
        tex: {
          inlineMath: [["\\\\(", "\\\\)"]],
          displayMath: [["\\\\[", "\\\\]"]],
          macros: {
            R: "\\\\mathbb{R}",
            N: "\\\\mathbb{N}",
            Borel: "\\\\mathcal{B}",
            Prob: "\\\\mathcal{P}",
            Pp: ["\\\\mathcal{P}_{#1}", 1],
            Couplings: "\\\\Pi",
            Lip: "\\\\mathrm{Lip}",
            supp: "\\\\operatorname{supp}",
            esssup: "\\\\operatorname*{ess\\\\,sup}",
            tr: "\\\\operatorname{tr}",
            diag: "\\\\operatorname{diag}",
            rank: "\\\\operatorname{rank}",
            Normal: "\\\\mathcal{N}",
            Id: "\\\\mathrm{Id}",
            Wass: "W",
            d: "\\\\mathrm{d}",
            abs: ["\\\\lvert #1\\\\rvert", 1],
            norm: ["\\\\lVert #1\\\\rVert", 1],
            inner: ["\\\\langle #1,\\\\,#2\\\\rangle", 2],
          }
        },
        svg: { fontCache: "global" }
      };
    </script>`;
}

function siteHeader(sections, currentSection) {
  const curOut = outPathOf(currentSection);
  const mainSecs = sections.filter((s) => s.data.group !== "appendix");
  const appendixSecs = sections.filter((s) => s.data.group === "appendix");

  const renderLink = (s, label) => {
    const cls = s.data.id === currentSection.data.id ? " is-current" : "";
    const href = relUrl(curOut, outPathOf(s));
    return `          <a href="${escapeHtml(href)}" class="site-header__link${cls}"><span class="site-header__num">${label}</span>${escapeHtml(s.data.nav ?? s.data.title)}</a>`;
  };

  const mainLinks = mainSecs.map((s, i) => renderLink(s, i + 1)).join("\n");
  const appendixLinks = appendixSecs
    .map((s, i) => renderLink(s, String.fromCharCode(65 + i)))
    .join("\n");

  const appendixNav = appendixSecs.length
    ? `\n          <span class="site-header__group-label">付録</span>\n${appendixLinks}`
    : "";

  const graphHref = relUrl(curOut, "graph.html");
  return `<header class="site-header">
      <div class="site-header__inner">
        <a href="${escapeHtml(relUrl(curOut, "index.html"))}" class="site-header__home">
          <span class="site-header__logo">OT</span>
          <span class="site-header__name">最適輸送問題</span>
        </a>
        <nav class="site-header__nav">
${mainLinks}${appendixNav}
          <a href="${escapeHtml(graphHref)}" class="site-header__link site-header__graph-link"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="18" r="2.5"/><line x1="8.5" y1="6" x2="15.5" y2="6"/><line x1="6" y1="8.5" x2="6" y2="15.5"/><line x1="8" y1="8" x2="16" y2="16"/></svg>グラフ</a>
        </nav>
      </div>
    </header>`;
}

function chapterTemplate(section, sections, index) {
  const curOut = outPathOf(section);
  // 参照ジャンプ用に「現在ページからの相対 URL」で章ファイル表を作る（app.js が使用）。
  const chapterFilesMap = {};
  sections.forEach((s) => {
    chapterFilesMap[s.data.id] = relUrl(curOut, outPathOf(s));
  });
  const chapterNum = String(index + 1).padStart(2, "0");
  const eyebrow = section.data.eyebrow
    ? `<p class="chapter-hero__eyebrow">${escapeHtml(section.data.eyebrow)}</p>`
    : "";

  let pagerPrev = "<span></span>";
  let pagerNext = "<span></span>";
  if (index > 0) {
    const p = sections[index - 1];
    pagerPrev = `<a class="chapter-pager__link chapter-pager__prev" href="${escapeHtml(relUrl(curOut, outPathOf(p)))}">
          <span class="chapter-pager__dir">&larr; 前の章</span>
          <strong>${escapeHtml(p.data.title)}</strong>
        </a>`;
  }
  if (index < sections.length - 1) {
    const n = sections[index + 1];
    pagerNext = `<a class="chapter-pager__link chapter-pager__next" href="${escapeHtml(relUrl(curOut, outPathOf(n)))}">
          <span class="chapter-pager__dir">次の章 &rarr;</span>
          <strong>${escapeHtml(n.data.title)}</strong>
        </a>`;
  }

  const graphBase = relUrl(curOut, "graph.html");
  const blocksJson = JSON.stringify(allBlocks).replaceAll("__GRAPH_BASE__", graphBase).replace(/<\//g, "<\\/");
  const filesJson = JSON.stringify(chapterFilesMap);

  return `<!doctype html>
<!-- Generated by build.mjs — edit site/content/*.md instead -->
<html lang="ja">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${escapeHtml(section.data.title)} — 最適輸送問題</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="${relUrl(curOut, "styles.css")}" />
    ${mathJaxScript()}
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
      window.__blocks = ${blocksJson};
      window.__chapterFiles = ${filesJson};
      window.__currentChapter = "${escapeHtml(section.data.id)}";
    </script>
    <script defer src="${relUrl(curOut, "app.js")}"></script>
  </head>
  <body>
    <div class="reading-progress" aria-hidden="true">
      <div class="reading-progress__fill"></div>
    </div>

    ${siteHeader(sections, section)}

    <div class="page-layout">
      <nav class="chapter-toc" aria-label="目次"></nav>

      <main class="content">
        <div class="chapter-hero">
          <span class="chapter-hero__num" aria-hidden="true">${chapterNum}</span>
          <div class="chapter-hero__text">
            ${eyebrow}
            <h1 class="chapter-hero__title">${escapeHtml(section.data.title)}</h1>
          </div>
        </div>

        <article class="prose" id="${escapeHtml(section.data.id)}">
          ${section.html.replaceAll("__GRAPH_BASE__", relUrl(curOut, "graph.html"))}
        </article>

        <nav class="chapter-pager">
          ${pagerPrev}
          ${pagerNext}
        </nav>
      </main>

      <aside class="ref-sidebar" aria-label="参照">
        <div class="ref-sidebar__header">
          <span class="ref-sidebar__label">参照</span>
        </div>
        <div class="ref-sidebar__body">
          <p class="ref-sidebar__empty">参照リンクをクリックすると<br>ここに定義や定理が表示されます</p>
        </div>
      </aside>
    </div>

    <dialog class="ref-sheet" id="ref-sheet">
      <div class="ref-sheet__content"></div>
      <button class="ref-sheet__close" type="button" aria-label="閉じる">&times;</button>
    </dialog>
  </body>
</html>
`;
}

function landingTemplate(sections) {
  const renderCards = (secs) =>
    secs
      .map((s, i) => {
        const num =
          s.data.group === "appendix"
            ? String.fromCharCode(65 + i)
            : String(i + 1).padStart(2, "0");
        const eyebrow = s.data.eyebrow
          ? `\n          <span class="toc-card__eyebrow">${escapeHtml(s.data.eyebrow)}</span>`
          : "";
        return `        <a href="${escapeHtml(outPathOf(s))}" class="toc-card">
          <span class="toc-card__num">${num}</span>${eyebrow}
          <h2 class="toc-card__title">${escapeHtml(s.data.title)}</h2>
        </a>`;
      })
      .join("\n");

  const mainSecs = sections.filter((s) => s.data.group !== "appendix");
  const appendixSecs = sections.filter((s) => s.data.group === "appendix");

  const appendixBlock = appendixSecs.length
    ? `
      <div class="landing__group">
        <h2 class="landing__group-title">付録：前提知識</h2>
        <p class="landing__group-sub">発表では省略した数学的前提を網羅した完全版．本編から参照される．</p>
      </div>
      <nav class="landing__toc">
${renderCards(appendixSecs)}
      </nav>`
    : "";

  return `<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Wasserstein 距離セミナー</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="./styles.css" />
  </head>
  <body>
    <main class="landing">
      <div class="landing__hero">
        <h1 class="landing__title">
          最適輸送問題
          <span>セミナー資料</span>
        </h1>
        <p class="landing__sub">Wasserstein 距離の理論と性質</p>
      </div>
      <nav class="landing__toc">
${renderCards(mainSecs)}
      </nav>${appendixBlock}
      <div class="landing__group">
        <h2 class="landing__group-title">ツール</h2>
      </div>
      <nav class="landing__toc landing__toc--tools">
        <a href="graph.html" class="toc-card toc-card--graph" style="animation-delay:0ms">
          <span class="toc-card__num"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="18" r="2.5"/><line x1="8.5" y1="6" x2="15.5" y2="6"/><line x1="6" y1="8.5" x2="6" y2="15.5"/><line x1="8" y1="8" x2="16" y2="16"/></svg></span>
          <span class="toc-card__eyebrow">INTERACTIVE</span>
          <h2 class="toc-card__title">依存グラフ</h2>
          <span class="toc-card__stats">${allBlocks.length} ブロックの関係を可視化</span>
        </a>
      </nav>

      <footer class="landing__footer">
        <p>参考文献: Givens–Shortt (1984), Villani (2009), Peyré–Cuturi (2019), 桑江ほか (2015)</p>
      </footer>
    </main>
  </body>
</html>
`;
}

const GRAPH_TYPES = new Set(["definition", "theorem", "proposition", "lemma"]);

function buildGraphData() {
  const graphBlocks = allBlocks.filter(b => GRAPH_TYPES.has(b.type));
  const graphIds = new Set(graphBlocks.map(b => b.id));
  const blocksByName = {};
  for (const block of graphBlocks) {
    blocksByName[block.name] = block;
  }
  const edges = [];
  for (const block of graphBlocks) {
    const refs = [...block.html.matchAll(/data-ref="([^"]+)"/g)].map(m => m[1]);
    for (const refName of new Set(refs)) {
      const target = blocksByName[refName];
      if (target && target.id !== block.id) {
        edges.push({ from: block.id, to: target.id });
      }
    }
  }
  return {
    nodes: graphBlocks.map(b => ({
      id: b.id,
      name: b.name,
      type: b.type,
      title: b.title,
      chapter: b.chapter,
      html: b.html,
    })),
    edges,
  };
}

function graphTemplate(sections) {
  const chapterFilesMap = {};
  sections.forEach((s) => {
    chapterFilesMap[s.data.id] = outPathOf(s);
  });

  const mainSecs = sections.filter((s) => s.data.group !== "appendix");
  const appendixSecs = sections.filter((s) => s.data.group === "appendix");
  const navLinks = (secs, labelFn) =>
    secs.map((s, i) => {
      const href = outPathOf(s);
      return `          <a href="${escapeHtml(href)}" class="site-header__link"><span class="site-header__num">${labelFn(i)}</span>${escapeHtml(s.data.nav ?? s.data.title)}</a>`;
    }).join("\n");
  const appendixNav = appendixSecs.length
    ? `\n          <span class="site-header__group-label">付録</span>\n${navLinks(appendixSecs, i => String.fromCharCode(65 + i))}`
    : "";

  return `<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>依存グラフ — Wasserstein 距離</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="./styles.css" />
    ${mathJaxScript()}
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.2/cytoscape.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
    <script>
      window.__chapterFiles = ${JSON.stringify(chapterFilesMap)};
      window.__graphData = ${JSON.stringify(graphData).replace(/<\//g, "<\\/")};
    </script>
    <script defer src="./graph.js"></script>
  </head>
  <body class="graph-body">
    <header class="site-header">
      <div class="site-header__inner">
        <a href="index.html" class="site-header__home">
          <span class="site-header__logo">OT</span>
          <span class="site-header__name">最適輸送問題</span>
        </a>
        <nav class="site-header__nav">
${navLinks(mainSecs, i => i + 1)}${appendixNav}
          <a href="graph.html" class="site-header__link is-current"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="18" r="2.5"/><line x1="8.5" y1="6" x2="15.5" y2="6"/><line x1="6" y1="8.5" x2="6" y2="15.5"/><line x1="8" y1="8" x2="16" y2="16"/></svg>グラフ</a>
        </nav>
      </div>
    </header>

    <div class="graph-layout">
      <aside class="graph-sidebar" id="graph-sidebar">
        <div class="graph-controls">
          <h3 class="graph-controls__title">フィルタ</h3>
          <label class="graph-controls__label">
            章
            <select id="chapter-filter" class="graph-controls__select">
              <option value="all">すべて</option>
            </select>
          </label>
          <label class="graph-controls__label">
            タイプ
            <select id="type-filter" class="graph-controls__select">
              <option value="all">すべて</option>
              <option value="definition">定義</option>
              <option value="theorem">定理</option>
              <option value="proposition">命題</option>
              <option value="lemma">補題</option>
            </select>
          </label>
          <label class="graph-controls__label">
            <input type="checkbox" id="hide-isolated"> 孤立ノードを非表示
          </label>
          <button id="fit-btn" class="graph-controls__btn" type="button">全体表示</button>
        </div>

        <div class="graph-legend">
          <h3 class="graph-legend__title">凡例</h3>
          <div class="graph-legend__items">
            <span class="graph-legend__item"><span class="graph-legend__dot" style="background:var(--teal)"></span>定義</span>
            <span class="graph-legend__item"><span class="graph-legend__dot" style="background:var(--indigo)"></span>定理</span>
            <span class="graph-legend__item"><span class="graph-legend__dot" style="background:var(--orange)"></span>命題</span>
            <span class="graph-legend__item"><span class="graph-legend__dot" style="background:#a855f7"></span>補題</span>
          </div>
        </div>

        <div class="graph-detail" id="graph-detail">
          <p class="graph-detail__empty">ノードをクリックすると<br>詳細が表示されます</p>
        </div>
      </aside>
      <div class="graph-main">
        <div class="focus-bar" id="focus-bar">
          <span class="focus-bar__hint">ダブルクリックでフォーカスモード</span>
        </div>
        <main class="graph-canvas" id="cy"></main>
      </div>
    </div>
  </body>
</html>
`;
}

/* ==========================================================================
   Build
   ========================================================================== */

const sections = readdirSync(contentDir)
  .filter((file) => file.endsWith(".md"))
  .sort()
  .map((file) => {
    const fullPath = path.join(contentDir, file);
    const source = readFileSync(fullPath, "utf8");
    const parsed = parseFrontmatter(source, fullPath);
    currentChapterId = parsed.data.id || null;
    return {
      file,
      data: parsed.data,
      html: renderMarkdown(parsed.body)
    };
  });

for (const section of sections) {
  for (const key of ["id", "title", "nav"]) {
    if (!section.data[key]) {
      throw new Error(`${section.file}: missing frontmatter key "${key}"`);
    }
  }
}

rmSync(distDir, { recursive: true, force: true });
mkdirSync(path.join(distDir, "main"), { recursive: true });
mkdirSync(path.join(distDir, "appendix"), { recursive: true });

writeFileSync(path.join(distDir, "index.html"), landingTemplate(sections), "utf8");
sections.forEach((section, i) => {
  writeFileSync(path.join(distDir, outPathOf(section)), chapterTemplate(section, sections, i), "utf8");
});

// グラフデータとグラフページを生成する。
const graphData = buildGraphData();
writeFileSync(path.join(distDir, "graph-data.json"), JSON.stringify(graphData), "utf8");
writeFileSync(path.join(distDir, "graph.html"), graphTemplate(sections), "utf8");

// html から相対参照される静的アセットを dist へコピーする。
for (const asset of ["styles.css", "app.js", "graph.js"]) {
  const src = path.join(siteRoot, asset);
  if (existsSync(src)) copyFileSync(src, path.join(distDir, asset));
}

const nMain = sections.filter((s) => s.data.group !== "appendix").length;
console.log(
  `Built dist/ : index.html + graph.html + main/(${nMain}) + appendix/(${sections.length - nMain}) — ${allBlocks.length} blocks, ${graphData.edges.length} edges.`
);
