#!/usr/bin/env node
// site/content/*.md → site/dist/*.html
//
//   node tools/site/build.mjs <セミナーのディレクトリ>
//
// 例: node tools/site/build.mjs seminar/cuturi
//
// dist/ は毎回作り直す。数式マクロは tex/preamble.tex から自動抽出するので、
// MathJax 用に別途書き写す必要はない。

import { readdirSync, readFileSync, writeFileSync, mkdirSync, copyFileSync, rmSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { loadConfig } from "./lib/config.mjs";
import { MarkdownRenderer } from "./lib/markdown.mjs";
import { buildGraphData } from "./lib/graph.mjs";
import { resolveMacros } from "./lib/macros.mjs";
import { chapterTemplate, landingTemplate, graphTemplate, outPathOf } from "./lib/templates.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ASSETS_DIR = path.join(__dirname, "assets");
const ASSETS = ["styles.css", "app.js", "graph.js"];

const seminarDir = process.argv.slice(2).find((a) => !a.startsWith("--"));
if (!seminarDir) {
  console.error("usage: node tools/site/build.mjs <セミナーのディレクトリ>");
  process.exit(2);
}

const config = await loadConfig(seminarDir);

// --- frontmatter ---

function parseFrontmatter(source, filePath) {
  const src = source.replaceAll("\r\n", "\n");
  if (!src.startsWith("---\n")) {
    throw new Error(`${filePath}: frontmatter is required`);
  }
  const end = src.indexOf("\n---\n", 4);
  if (end === -1) {
    throw new Error(`${filePath}: frontmatter is not closed`);
  }
  const data = {};
  for (const line of src.slice(4, end).split("\n")) {
    if (!line.trim()) continue;
    const index = line.indexOf(":");
    if (index === -1) {
      throw new Error(`${filePath}: invalid frontmatter line "${line}"`);
    }
    data[line.slice(0, index).trim()] = line.slice(index + 1).trim();
  }
  return { data, body: src.slice(end + 5) };
}

// --- content/*.md → HTML ---

if (!existsSync(config.contentDir)) {
  throw new Error(
    `${config.contentDir} がない。先に tools/site/tex2md.mjs を実行すること。`
  );
}

const renderer = new MarkdownRenderer(config);

const sections = readdirSync(config.contentDir)
  .filter((file) => file.endsWith(".md"))
  .sort()
  .map((file) => {
    const fullPath = path.join(config.contentDir, file);
    const parsed = parseFrontmatter(readFileSync(fullPath, "utf8"), fullPath);
    return {
      file,
      data: parsed.data,
      html: renderer.render(parsed.body, parsed.data.id || null),
    };
  });

if (sections.length === 0) {
  throw new Error(`${config.contentDir} に *.md がない。先に tex2md.mjs を実行すること。`);
}

for (const section of sections) {
  for (const key of ["id", "title", "nav"]) {
    if (!section.data[key]) {
      throw new Error(`${section.file}: missing frontmatter key "${key}"`);
    }
  }
}

// --- 数式マクロ（preamble.tex を唯一の定義元にする）---

const macroWarnings = [];
const macros = resolveMacros(config.preamblePath, {
  overrides: config.macroOverrides,
  ignore: config.macroIgnore,
  warn: (m) => macroWarnings.push(m),
});
for (const w of macroWarnings) console.warn(`  ⚠ macro: ${w}`);

// --- 出力 ---

const blocks = renderer.blocks;
const graphData = buildGraphData(blocks, config.graphTypes);

rmSync(config.distDir, { recursive: true, force: true });
mkdirSync(path.join(config.distDir, "main"), { recursive: true });
mkdirSync(path.join(config.distDir, "appendix"), { recursive: true });

writeFileSync(
  path.join(config.distDir, "index.html"),
  landingTemplate(config, sections, blocks),
  "utf8"
);
sections.forEach((section, i) => {
  writeFileSync(
    path.join(config.distDir, outPathOf(section)),
    chapterTemplate(config, section, sections, i, blocks, macros),
    "utf8"
  );
});

writeFileSync(
  path.join(config.distDir, "graph-data.json"),
  JSON.stringify(graphData),
  "utf8"
);
writeFileSync(
  path.join(config.distDir, "graph.html"),
  graphTemplate(config, sections, graphData, macros),
  "utf8"
);

// html から相対参照される静的アセットを dist へコピーする。
// セミナー側に同名ファイルがあればそちらを優先し、個別の上書きを可能にする。
for (const asset of ASSETS) {
  const override = path.join(config.seminarDir, "site-assets", asset);
  const src = existsSync(override) ? override : path.join(ASSETS_DIR, asset);
  copyFileSync(src, path.join(config.distDir, asset));
}

const nMain = sections.filter((s) => s.data.group !== "appendix").length;
console.log(
  `Built dist/ : index.html + graph.html + main/(${nMain}) + appendix/(${sections.length - nMain}) — ` +
  `${blocks.length} blocks, ${graphData.edges.length} edges, ${Object.keys(macros).length} macros.`
);
