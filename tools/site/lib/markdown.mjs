// content/*.md（tex2md が出す中間 markdown）を HTML に変換する。
//
// 対応する記法は素の markdown ではなく、この資料専用の方言：
//   - 見出し ## / ### / ####、箇条書き、番号付きリスト、コードフェンス
//   - \( \) と \[ \] の数式（MathJax がそのまま描画する）
//   - ::: で開閉するコンテナ（定義・定理・証明の折りたたみ など）
//   - [ref:表示|参照名] と [term:表示|用語 id] のリンク

import { BLOCK_TYPES, ID_PREFIX } from "./blocks.mjs";

export function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

// ::: の指定 → 開きタグ・閉じタグ・記録するブロック種別。
// cssClass が null の種別（注意・例）は専用のガワを持つ。
function containerSpecs() {
  const specs = {
    "grid two": { open: '<div class="grid two">', close: "</div>" },
    compare: { open: '<div class="compare">', close: "</div>" },
    column: { open: "<div>", close: "</div>" },
    fact: { open: '<aside class="margin-note">', close: "</aside>", type: "remark" },
    "fact accent": {
      open: '<div class="example-band"><article class="example-band__inner">',
      close: "</article></div>",
      type: "example",
    },
  };
  for (const [type, meta] of Object.entries(BLOCK_TYPES)) {
    if (!meta.cssClass) continue;
    specs[type] = {
      open: `<div class="block ${meta.cssClass}">`,
      close: "</div>",
      type,
    };
  }
  return specs;
}

const CONTAINERS = containerSpecs();

// 見出しから参照キーを取り出す正規表現。
// 「Lem 2.2.3: タイトル」（番号付き）と「Lem: タイトル」の両形式を許す。
const HEADING_PREFIXES = Object.values(ID_PREFIX)
  .map((p) => p[0].toUpperCase() + p.slice(1))
  .join("|");
const HEADING_NAME_RE = new RegExp(
  `^(?:${HEADING_PREFIXES})(?:\\s+[0-9A-Z.]+)?:\\s*(.+)$`,
);
const REF_DISPLAY_RE = new RegExp(
  `^(${HEADING_PREFIXES})(\\s+[0-9A-Z.]+)?:\\s*(.+)$`,
);

export class MarkdownRenderer {
  constructor(config) {
    this.config = config;
    this.blocks = [];
    this.usedBlockIds = new Set();
    this.currentChapterId = null;
  }

  makeBlockId(type, rawName) {
    const prefix = ID_PREFIX[type] || type;
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
    if (this.usedBlockIds.has(id)) {
      let n = 2;
      while (this.usedBlockIds.has(`${id}-${n}`)) n += 1;
      id = `${id}-${n}`;
    }
    this.usedBlockIds.add(id);
    return id;
  }

  // 行内記法（強調・数式・参照リンク）を HTML にする。
  renderInline(source) {
    const mathSpans = [];
    const shielded = source.replace(/\\\([^]*?\\\)/g, (m) => {
      mathSpans.push(m);
      return `\x00MATH${mathSpans.length - 1}\x00`;
    });

    const result = escapeHtml(shielded)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(
        /\[term:([^|\]]+)\|([a-z0-9-]+)\]/g,
        (_m, label, term) =>
          `<button type="button" class="term" data-term="${term}">${label}</button>`
      )
      .replace(/\[ref:([^|\]]+?)(?:\|([^\]]+))?\]/g, (_m, first, second) => {
        const refName = second || first;
        // 旧形式「Lem: タイトル」は種別だけに短縮。新形式「補題 2.2.3」はそのまま。
        // ツールチップには参照先のタイトルを出す。
        const typeMatch = REF_DISPLAY_RE.exec(first);
        const display = typeMatch ? `${typeMatch[1]}${typeMatch[2] ?? ""}` : first;
        return `<button type="button" class="ref" data-ref="${escapeHtml(refName)}" title="${escapeHtml(refName)}">${escapeHtml(display)}</button>`;
      });

    return result.replace(/\x00MATH(\d+)\x00/g, (_, i) =>
      escapeHtml(mathSpans[parseInt(i, 10)])
    );
  }

  render(markdown, chapterId) {
    this.currentChapterId = chapterId ?? null;
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
      html.push(`<p>${this.renderInline(paragraph.join(" "))}</p>`);
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
        throw new Error("対応する ::: のないコンテナ終了");
      }
      if (currentBlock && stack.length === currentBlock.depth) {
        if (currentBlock.name) {
          this.blocks.push({
            id: currentBlock.id,
            name: currentBlock.name,
            type: currentBlock.type,
            title: currentBlock.fullTitle,
            chapter: this.currentChapterId,
            html: html.slice(currentBlock.divIndex + 1).join("\n"),
          });
        }
        currentBlock = null;
      }
      html.push(closing);
    };

    const openContainer = (spec) => {
      flushParagraph();
      closeList();

      if (spec.startsWith("details-embedded ")) {
        const title = spec.slice("details-embedded ".length);
        html.push(`<details class="proof"><summary>${this.renderInline(title)}</summary>`);
        stack.push("</details>");
        return;
      }
      if (spec.startsWith("details ")) {
        const title = spec.slice("details ".length);
        html.push(`<details class="fold"><summary>${this.renderInline(title)}</summary>`);
        stack.push("</details>");
        return;
      }
      if (spec.startsWith("demo ")) {
        // セミナー固有の図。site.config.mjs の demos から差し込む。
        const name = spec.slice("demo ".length).trim();
        const demo = this.config.demos[name];
        if (!demo) throw new Error(`未定義のデモ図 "${name}"（site.config.mjs の demos を確認）`);
        html.push(typeof demo === "function" ? demo() : demo);
        return;
      }

      const container = CONTAINERS[spec];
      if (!container) throw new Error(`未知のコンテナ ":::${spec}"`);
      if (container.type) {
        currentBlock = { type: container.type, divIndex: html.length, depth: stack.length };
      }
      html.push(container.open);
      stack.push(container.close);
    };

    for (let i = 0; i < lines.length; i += 1) {
      const raw = lines[i];
      const trimmed = raw.trim();

      if (trimmed === "") {
        flushParagraph();
        // 空行を挟んで同種のリストが続く場合は 1 つのリストとして扱う。
        let nextIndex = i + 1;
        while (nextIndex < lines.length && lines[nextIndex].trim() === "") nextIndex += 1;
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
        if (i >= lines.length) throw new Error("閉じられていないコードフェンス (```)");
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
          const nameMatch = HEADING_NAME_RE.exec(rawTitle);
          const name = nameMatch ? nameMatch[1].trim() : rawTitle.trim();
          const id = this.makeBlockId(currentBlock.type, name);
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
          html.push(`<h2 id="sec-${slug}">${this.renderInline(heading[2])}</h2>`);
        } else if (
          level === 3 &&
          currentBlock &&
          this.config.graphTypes.has(currentBlock.type) &&
          currentBlock.id
        ) {
          html.push(`<h3>${this.renderInline(heading[2])}${graphLink(currentBlock.id)}</h3>`);
        } else {
          html.push(`<h${level}>${this.renderInline(heading[2])}</h${level}>`);
        }
        continue;
      }

      const ordered = /^\d+\.\s+(.+)$/.exec(trimmed);
      if (ordered) {
        flushParagraph();
        openList("ol");
        html.push(`<li>${this.renderInline(ordered[1])}</li>`);
        continue;
      }

      const unordered = /^[-*]\s+(.+)$/.exec(trimmed);
      if (unordered) {
        flushParagraph();
        openList("ul");
        html.push(`<li>${this.renderInline(unordered[1])}</li>`);
        continue;
      }

      paragraph.push(trimmed);
    }

    flushParagraph();
    closeList();
    if (stack.length > 0) {
      throw new Error(`${stack.length} 個の ::: が閉じられていない`);
    }

    return html.join("\n");
  }
}

// 依存グラフへのリンク。__GRAPH_BASE__ は出力時にページ相対 URL へ差し替える。
function graphLink(blockId) {
  return `<a class="block__graph-link" href="__GRAPH_BASE__?focus=${encodeURIComponent(blockId)}" title="依存グラフで表示"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="5" r="2.5"/><circle cx="5" cy="19" r="2.5"/><circle cx="19" cy="19" r="2.5"/><line x1="12" y1="7.5" x2="5" y2="16.5"/><line x1="12" y1="7.5" x2="19" y2="16.5"/></svg></a>`;
}
