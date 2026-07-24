// 定義・定理どうしの参照関係から依存グラフのデータを組む。
//
// 辺は「ブロック A の本文が [ref:…] でブロック B を参照している」= A は B に依存する、
// の向きで A -> B を張る。

// data-ref 属性は HTML エスケープ済みなので、ブロック名と突き合わせる前に戻す。
function unescapeHtml(value) {
  return value
    .replaceAll("&quot;", '"')
    .replaceAll("&gt;", ">")
    .replaceAll("&lt;", "<")
    .replaceAll("&amp;", "&");
}

export function buildGraphData(blocks, graphTypes) {
  const graphBlocks = blocks.filter((b) => graphTypes.has(b.type));
  const blocksByName = {};
  for (const block of graphBlocks) blocksByName[block.name] = block;

  const edges = [];
  for (const block of graphBlocks) {
    const refs = [...block.html.matchAll(/data-ref="([^"]+)"/g)].map((m) =>
      unescapeHtml(m[1])
    );
    for (const refName of new Set(refs)) {
      const target = blocksByName[refName];
      if (target && target.id !== block.id) {
        edges.push({ from: block.id, to: target.id });
      }
    }
  }

  return {
    nodes: graphBlocks.map((b) => ({
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
