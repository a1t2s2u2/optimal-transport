'use strict';

const ENV_COLOR = {
  definition: '#2563eb', proposition: '#ea7317', theorem: '#dc2626',
  claim: '#16a34a', remark: '#ca8a04', example: '#6b7280',
};
const ENV_LABEL = {
  definition: 'Def', proposition: 'Prop', theorem: 'Thm',
  claim: 'Clm', remark: 'Rem', example: 'Ex',
};
const TINT = ['#dbeafe', '#dcfce7', '#fef3c7', '#fae8ff',
              '#ffedd5', '#cffafe', '#fee2e2', '#ede9fe', '#f1f5f9', '#fce7f3'];
const SNIPPET_LEN = 80;
const TO_DETAIL = 1.7;    // 概要のフィットズーム×これ以上にズームインで詳細へ
const TO_OVERVIEW = 0.55; // 詳細のフィットズーム×これ以下にズームアウトで概要へ

let GRAPH = null;
let cy = null;
let NODE_BY_ID = {};
let CONCEPTS = [];          // 概念名（出現順）
let UP = {}, DOWN = {};
let activeRouteByNode = {};
let focusSet = null;
let selectedId = null;
let mode = 'overview';      // 'overview'（概念カード）| 'detail'（全ブロック）
let ovZoom = 1, dtZoom = 1;
let lodPending = false;

const DATA_SRC = new URLSearchParams(location.search).get('data') || '../out/graph.json';

function conceptOf(n) { return n.group || n.chapter || 'その他'; }
function conId(c) { return 'con::' + c; }
function csId(c) { return 'cs::' + c; }
function tintOf(c) { const i = CONCEPTS.indexOf(c); return TINT[(i < 0 ? 0 : i) % TINT.length]; }

// ---------------------------------------------------------------------------
// 読み込み
// ---------------------------------------------------------------------------
async function load() {
  let data;
  try {
    const res = await fetch(DATA_SRC, { cache: 'no-store' });
    if (!res.ok) throw new Error(res.status + ' ' + res.statusText);
    data = await res.json();
  } catch (e) {
    showError(
      `データ（${DATA_SRC}）を読み込めませんでした（` + e.message + '）。\n\n' +
      'まず抽出してローカルサーバ経由で開いてください：\n\n' +
      '  cd proofgraph && uv run pg.py        # 抽出 → 検証 → サーバ起動\n' +
      '  → http://localhost:8000/viewer/\n\n' +
      'サンプル（抽出不要）を見るには:\n' +
      '  → http://localhost:8000/viewer/?data=sample.graph.json'
    );
    return;
  }
  GRAPH = data;
  GRAPH.nodes.forEach(n => { NODE_BY_ID[n.id] = n; });
  CONCEPTS = [...new Set(GRAPH.nodes.map(conceptOf))];
  buildAdjacency();
  buildControls();
  buildGraph();
  document.getElementById('stats').textContent =
    `${data.stats.n_nodes} blocks · ${data.stats.n_edges} edges · 概念 ${CONCEPTS.length}`;
}

function showError(msg) {
  const el = document.getElementById('error');
  el.hidden = false;
  el.textContent = msg;
}

function buildAdjacency() {
  GRAPH.nodes.forEach(n => { UP[n.id] = new Set(); DOWN[n.id] = new Set(); });
  GRAPH.edges.forEach(e => {
    if (!NODE_BY_ID[e.to] || !NODE_BY_ID[e.from]) return;
    UP[e.from].add(e.to);
    DOWN[e.to].add(e.from);
  });
}

// ---------------------------------------------------------------------------
// コントロール構築
// ---------------------------------------------------------------------------
function buildControls() {
  const sf = document.getElementById('space-filter');
  Object.keys(GRAPH.spaces).forEach(key => {
    const o = document.createElement('option');
    o.value = key;
    o.textContent = `${GRAPH.spaces[key].label}（${key}）まで`;
    sf.appendChild(o);
  });
  sf.addEventListener('change', updateView);
  document.getElementById('show-uses').addEventListener('change', updateView);
  document.getElementById('show-proof').addEventListener('change', updateView);
  document.getElementById('focus-scope').addEventListener('change', () => {
    if (selectedId) applyFocus(selectedId); else clearFocus();
  });
  document.getElementById('fit-btn').addEventListener('click', clearFocus);

  const curData = new URLSearchParams(location.search).get('data') || '';
  document.querySelectorAll('.dataset a').forEach(a => {
    const linkData = new URL(a.getAttribute('href'), location.href).searchParams.get('data') || '';
    a.classList.toggle('active', linkData === curData);
  });
}

// ---------------------------------------------------------------------------
// グラフ構築
// ---------------------------------------------------------------------------
function cardLabel(n) {
  const head = `${ENV_LABEL[n.env]}. ${mathToText(n.title) || stripMath(n.title)}`;
  const snip = mathToText(n.statement || '');
  const body = snip.length > SNIPPET_LEN ? snip.slice(0, SNIPPET_LEN) + '…' : snip;
  return body ? head + '\n' + body : head;
}

function buildGraph() {
  const elements = [];
  const cnt = {};
  GRAPH.nodes.forEach(n => { const c = conceptOf(n); cnt[c] = (cnt[c] || 0) + 1; });

  CONCEPTS.forEach(c => {
    elements.push({ data: { id: conId(c), label: c, tint: tintOf(c), con: c }, classes: 'concept' });
    elements.push({ data: { id: csId(c), label: `${c}\n（${cnt[c]} ブロック）`, tint: tintOf(c), con: c },
                    classes: 'csum' });
  });
  GRAPH.nodes.forEach(n => {
    elements.push({ data: {
      id: n.id, label: cardLabel(n), env: n.env, chapter: n.chapter, spaces: n.spaces,
      parent: conId(conceptOf(n)),
    }, classes: 'leaf' });
  });
  GRAPH.edges.forEach((e, i) => {
    if (!NODE_BY_ID[e.to]) return;
    elements.push({ data: {
      id: 'e' + i, source: e.from, target: e.to, kind: e.kind, route: e.route || '',
    }});
  });
  // 概念間の集約依存（概要図の矢印）。区切り文字を使わず入れ子マップで集計する。
  const cross = {};
  GRAPH.edges.forEach(e => {
    const a = NODE_BY_ID[e.from], b = NODE_BY_ID[e.to];
    if (!a || !b) return;
    const ca = conceptOf(a), cb = conceptOf(b);
    if (ca === cb) return;
    (cross[ca] = cross[ca] || {})[cb] = (cross[ca][cb] || 0) + 1;
  });
  CONCEPTS.forEach((a, ai) => {
    Object.keys(cross[a] || {}).forEach(b => {
      elements.push({ data: {
        id: 'ce::' + ai + '>>' + CONCEPTS.indexOf(b), source: csId(a), target: csId(b), weight: cross[a][b],
      }, classes: 'cedge' });
    });
  });

  cy = cytoscape({
    container: document.getElementById('cy'),
    elements,
    wheelSensitivity: 0.2,
    style: [
      { selector: 'node.leaf', style: {
        'shape': 'round-rectangle', 'background-color': '#ffffff', 'background-opacity': 1,
        'border-width': 2, 'border-color': ele => ENV_COLOR[ele.data('env')] || '#888',
        'label': 'data(label)', 'text-wrap': 'wrap', 'text-max-width': '180px',
        'text-valign': 'center', 'text-halign': 'center',
        'font-size': '11px', 'line-height': 1.3, 'color': '#0f172a',
        'width': 'label', 'height': 'label', 'padding': '8px',
      }},
      { selector: 'node.concept', style: {
        'background-color': 'data(tint)', 'background-opacity': 0.35,
        'border-width': 1, 'border-color': '#cbd5e1', 'border-style': 'dashed',
        'shape': 'roundrectangle', 'padding': '26px',
        'label': 'data(label)', 'text-valign': 'top', 'text-halign': 'center',
        'font-size': '18px', 'font-weight': 'bold', 'color': '#475569', 'text-margin-y': 2,
      }},
      // 概念サマリーカード（概要図の主役。大きい文字）
      { selector: 'node.csum', style: {
        'display': 'none', 'shape': 'round-rectangle',
        'background-color': 'data(tint)', 'background-opacity': 0.95,
        'border-width': 3, 'border-color': '#475569',
        'label': 'data(label)', 'text-wrap': 'wrap', 'text-max-width': '240px',
        'text-valign': 'center', 'text-halign': 'center',
        'font-size': '26px', 'font-weight': 'bold', 'color': '#1e293b',
        'width': 'label', 'height': 'label', 'padding': '26px',
      }},
      { selector: 'node.leaf.dim', style: { 'opacity': 0.12 }},
      { selector: 'node.leaf.sel', style: {
        'border-width': 4, 'border-color': '#0f172a', 'background-color': '#f8fafc' }},
      { selector: 'node.leaf.hl', style: { 'border-width': 3, 'border-color': '#0f172a' }},
      { selector: 'node.leaf.support', style: {
        'border-width': 3, 'border-color': '#16a34a', 'opacity': 1 }},
      { selector: 'node.leaf.faded', style: { 'opacity': 0.07 }},
      { selector: 'edge.faded', style: { 'opacity': 0.025 }},
      { selector: 'edge', style: {
        'width': 1.4, 'line-color': '#c4c4cc', 'target-arrow-color': '#c4c4cc',
        'target-arrow-shape': 'triangle', 'arrow-scale': 0.9, 'curve-style': 'bezier', 'opacity': 0.45,
      }},
      { selector: 'edge[kind = "proof"]', style: { 'line-style': 'dashed' }},
      // 概念間の集約依存
      { selector: 'edge.cedge', style: {
        'display': 'none', 'width': ele => 3 + Math.min(8, ele.data('weight')),
        'line-color': '#64748b', 'target-arrow-color': '#64748b', 'target-arrow-shape': 'triangle',
        'arrow-scale': 1.6, 'opacity': 0.6, 'curve-style': 'bezier',
      }},
      { selector: 'edge.dim', style: { 'opacity': 0.03 }},
      { selector: 'edge.up', style: {
        'line-color': '#2563eb', 'target-arrow-color': '#2563eb', 'width': 2.6, 'opacity': 0.95 }},
      { selector: 'edge.down', style: {
        'line-color': '#7c3aed', 'target-arrow-color': '#7c3aed', 'width': 2.6, 'opacity': 0.95 }},
      { selector: 'edge.hl', style: {
        'line-color': '#0f172a', 'target-arrow-color': '#0f172a', 'width': 2.8, 'opacity': 1 }},
      { selector: 'edge.route-on', style: {
        'line-color': '#dc2626', 'target-arrow-color': '#dc2626', 'width': 3, 'opacity': 1 }},
    ],
  });

  cy.on('tap', 'node.leaf', evt => selectNode(evt.target.id()));
  cy.on('tap', 'node.csum', evt => openConcept(evt.target.data('con')));
  cy.on('tap', evt => { if (evt.target === cy) clearSelection(); });
  cy.on('zoom', () => {
    if (focusSet || lodPending) return;
    lodPending = true;
    requestAnimationFrame(() => {
      lodPending = false;
      if (mode === 'overview' && cy.zoom() >= ovZoom * TO_DETAIL) enterDetail();
      else if (mode === 'detail' && cy.zoom() <= dtZoom * TO_OVERVIEW) enterOverview();
    });
  });

  layoutDetail();   // メンバーの位置を確定（クリック時のフィット用）
  enterOverview();  // 既定は概念カードの概要図
}

function notAux(el) { return !el.hasClass('csum') && !el.hasClass('cedge'); }

function layoutDetail() {
  cy.nodes('.leaf').style('display', 'element');
  cy.elements().filter(notAux).layout({
    name: 'dagre', animate: false, fit: false, rankDir: 'BT', nodeSep: 24, rankSep: 70,
  }).run();
}

function layoutOverview() {
  cy.collection().union(cy.nodes('.csum')).union(cy.edges('.cedge')).layout({
    name: 'dagre', animate: false, fit: false, rankDir: 'BT', nodeSep: 160, rankSep: 180,
  }).run();
}

function enterOverview() {
  mode = 'overview';
  layoutOverview();
  applyOverview();
  cy.fit(cy.nodes('.csum'), 50);
  ovZoom = cy.zoom();
}

function enterDetail(fitConcept) {
  mode = 'detail';
  layoutDetail();
  applyDetail();
  if (fitConcept) cy.fit(cy.nodes('.leaf').filter(n => conceptOf(NODE_BY_ID[n.id()]) === fitConcept), 50);
  else cy.fit(cy.elements().filter(el => el.hasClass('leaf') || el.hasClass('concept')), 36);
  dtZoom = fitConcept ? cy.zoom() / 1.5 : cy.zoom();   // fitConcept 時は基準を緩める
}

function openConcept(c) {
  enterDetail(c);
}

// ---------------------------------------------------------------------------
// 表示更新
// ---------------------------------------------------------------------------
function filterCtx() {
  const spaceKey = document.getElementById('space-filter').value;
  return {
    showUses: document.getElementById('show-uses').checked,
    showProof: document.getElementById('show-proof').checked,
    provided: spaceKey ? new Set(GRAPH.spaces[spaceKey].closure) : null,
  };
}

function setEdgeVis(e, showUses, showProof) {
  const okKind = (e.data('kind') === 'uses' && showUses) || (e.data('kind') === 'proof' && showProof);
  const ends = e.source().style('display') === 'element' && e.target().style('display') === 'element';
  e.style('display', (okKind && ends) ? 'element' : 'none');
}

function dimLeaf(n, provided) {
  const d = NODE_BY_ID[n.id()];
  const holds = !(provided && d.spaces.length) || d.spaces.every(s => provided.has(s));
  n.toggleClass('dim', n.style('display') === 'element' && provided && !holds);
}

function updateView() {
  if (!cy) return;
  if (focusSet) applyExplicit();
  else if (mode === 'detail') applyDetail();
  else applyOverview();
}

// 概要: 概念カード＋概念間依存のみ
function applyOverview() {
  const { provided } = filterCtx();
  cy.batch(() => {
    cy.nodes('.concept').style('display', 'none');
    cy.nodes('.csum').style('display', 'element');
    cy.nodes('.leaf').style('display', 'none');
    cy.edges().filter(e => !e.hasClass('cedge')).style('display', 'none');
    cy.edges('.cedge').forEach(e => {
      const ends = e.source().style('display') === 'element' && e.target().style('display') === 'element';
      e.style('display', ends ? 'element' : 'none');
    });
    void provided;
  });
}

// 詳細: 全ブロック（概念コンテナでグループ）
function applyDetail() {
  const { showUses, showProof, provided } = filterCtx();
  cy.batch(() => {
    cy.nodes('.csum').style('display', 'none');
    cy.edges('.cedge').style('display', 'none');
    cy.nodes('.leaf').forEach(n => {
      const visible = !(focusSet && !focusSet.has(n.id()));
      n.style('display', visible ? 'element' : 'none');
      dimLeaf(n, provided);
    });
    cy.edges().filter(e => !e.hasClass('cedge')).forEach(e => setEdgeVis(e, showUses, showProof));
    cy.nodes('.concept').forEach(p =>
      p.style('display', p.children().some(c => c.style('display') === 'element') ? 'element' : 'none'));
  });
}

// フォーカス中（詳細の一種）
function applyExplicit() { applyDetail(); }

// ---------------------------------------------------------------------------
// フォーカス
// ---------------------------------------------------------------------------
function closure(id, adj, transitive) {
  const out = new Set();
  const stack = [...adj[id]];
  while (stack.length) {
    const x = stack.pop();
    if (out.has(x)) continue;
    out.add(x);
    if (transitive) adj[x].forEach(y => stack.push(y));
  }
  return out;
}

function applyFocus(id) {
  const scope = document.getElementById('focus-scope').value;
  if (scope === 'off') { clearFocus(); return; }
  const trans = scope === 'transitive';
  const up = closure(id, UP, trans);
  const down = closure(id, DOWN, trans);
  focusSet = new Set([id, ...up, ...down]);
  mode = 'detail';
  cy.nodes('.leaf').style('display', 'element');
  applyDetail();
  cy.batch(() => {
    cy.edges().removeClass('up down');
    cy.edges(':visible').forEach(e => {
      const s = e.source().id(), t = e.target().id();
      if (focusSet.has(s) && focusSet.has(t)) {
        if (up.has(t) || t === id) e.addClass('up');
        else if (down.has(s) || s === id) e.addClass('down');
      }
    });
  });
  cy.elements(':visible').filter(notAux).layout({
    name: 'dagre', animate: false, fit: true, padding: 36, rankDir: 'BT', nodeSep: 24, rankSep: 70,
  }).run();
  const n = NODE_BY_ID[id];
  document.getElementById('stats').textContent =
    `フォーカス: ${ENV_LABEL[n.env]}. ${stripMath(n.title)} — 依存 ${up.size} / 被依存 ${down.size}` +
    `（${trans ? '推移閉包' : '直接'}）`;
}

function clearFocus() {
  if (!cy) return;
  focusSet = null;
  cy.edges().removeClass('up down');
  enterOverview();
  document.getElementById('stats').textContent =
    `${GRAPH.stats.n_nodes} blocks · ${GRAPH.stats.n_edges} edges · 概念 ${CONCEPTS.length}`;
}

// ---------------------------------------------------------------------------
// 選択 & 詳細パネル
// ---------------------------------------------------------------------------
function selectNode(id) {
  clearSelection(false);
  const d = NODE_BY_ID[id];
  if (!d) return;
  selectedId = id;
  const node = cy.getElementById(id);
  node.addClass('sel');
  node.outgoers('edge').addClass('hl');
  node.outgoers('node').addClass('hl');
  node.incomers('edge').addClass('hl');
  node.incomers('node').addClass('hl');

  renderDetail(d);
  const routes = routesOf(id);
  const defaultRoute = activeRouteByNode[id] || (routes[0] && routes[0].route);
  highlightRoute(d, defaultRoute);

  if (document.getElementById('focus-scope').value !== 'off') applyFocus(id);
}

function clearSelection(clearPanel = true) {
  selectedId = null;
  cy.elements().removeClass('sel hl route-on support faded');
  if (clearPanel) {
    document.getElementById('detail-empty').hidden = false;
    document.getElementById('detail-body').hidden = true;
  }
}

function routesOf(id) { return GRAPH.routes.filter(r => r.node === id); }

function renderDetail(d) {
  document.getElementById('detail-empty').hidden = true;
  const body = document.getElementById('detail-body');
  body.hidden = false;

  const envEl = document.getElementById('d-env');
  envEl.className = 'badge ' + d.env;
  envEl.textContent = ENV_LABEL[d.env];

  document.getElementById('d-title').innerHTML = d.title;
  const loc = [d.group || d.chapter, d.section, d.subsection].filter(Boolean).map(stripMath).join(' › ');
  document.getElementById('d-loc').textContent = loc + '  ·  ' + d.id;

  document.getElementById('d-statement').innerHTML = texToHtml(d.statement);

  document.getElementById('d-spaces').innerHTML = d.spaces.length
    ? d.spaces.map(s => `<span class="chip">${GRAPH.spaces[s] ? GRAPH.spaces[s].label : s}</span>`).join('')
    : '<span class="muted">空間タグなし</span>';

  const fwrap = document.getElementById('d-focus');
  fwrap.innerHTML =
    `<button id="d-focus-direct">前後に絞る（直接）</button>` +
    `<button id="d-focus-trans">前後に絞る（推移）</button>`;
  document.getElementById('d-focus-direct').onclick = () => {
    document.getElementById('focus-scope').value = 'direct'; applyFocus(d.id);
  };
  document.getElementById('d-focus-trans').onclick = () => {
    document.getElementById('focus-scope').value = 'transitive'; applyFocus(d.id);
  };

  const routes = routesOf(d.id);
  const rwrap = document.getElementById('d-routes-wrap');
  const rdiv = document.getElementById('d-routes');
  rwrap.hidden = false;
  if (routes.length === 0) {
    rdiv.innerHTML = '<span class="muted">証明ルートなし（定義・公理的ブロック）</span>';
  } else {
    rdiv.innerHTML = '';
    routes.forEach(r => {
      const div = document.createElement('div');
      div.className = 'route';
      const name = r.route === '_auto' ? '自動抽出' : `ルート ${r.route}`;
      const deps = r.deps.length
        ? '<ul>' + r.deps.map(depLi).join('') + '</ul>'
        : '<span class="muted">（外部依存のみ／依存なし）</span>';
      div.innerHTML =
        `<div class="route-head">${name} ` +
        `<button data-route="${r.route}">辺を強調</button> ` +
        `<button data-support="${r.route}">支持集合</button></div>${deps}`;
      div.querySelector('[data-route]').addEventListener('click', () => {
        activeRouteByNode[d.id] = r.route; highlightRoute(d, r.route);
      });
      div.querySelector('[data-support]').addEventListener('click', () => {
        activeRouteByNode[d.id] = r.route; highlightSupport(d.id, r.route);
      });
      rdiv.appendChild(div);
    });
  }

  const allDeps = new Set();
  (d.uses || []).forEach(x => allDeps.add(x));
  routes.forEach(r => r.deps.forEach(x => allDeps.add(x)));
  GRAPH.edges.filter(e => e.from === d.id && e.kind === 'uses').forEach(e => allDeps.add(e.to));
  document.getElementById('d-deps').innerHTML =
    [...allDeps].map(depLi).join('') || '<li class="muted" style="cursor:default">なし</li>';

  const rdeps = GRAPH.edges.filter(e => e.to === d.id).map(e => e.from);
  document.getElementById('d-rdeps').innerHTML =
    [...new Set(rdeps)].map(depLi).join('') || '<li class="muted" style="cursor:default">なし</li>';

  if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([body]);
}

function depLi(id) {
  const t = NODE_BY_ID[id];
  if (!t) return `<li class="missing">${id}（未定義）</li>`;
  return `<li data-goto="${id}" onclick="window.__goto('${id}')">${ENV_LABEL[t.env]}. ${stripMath(t.title)}</li>`;
}
window.__goto = id => {
  if (!cy.getElementById(id).nonempty()) return;
  if (mode === 'overview') enterDetail();
  cy.animate({ center: { eles: cy.getElementById(id) }, zoom: Math.max(dtZoom * 1.4, 1) }, { duration: 300 });
  selectNode(id);
};

function highlightRoute(d, routeName) {
  cy.edges().removeClass('route-on');
  if (!routeName) return;
  const deps = (routesOf(d.id).find(r => r.route === routeName) || {}).deps || [];
  cy.edges().forEach(e => {
    if (e.data('kind') === 'proof' && e.source().id() === d.id &&
        e.data('route') === routeName && deps.includes(e.target().id())) {
      e.addClass('route-on');
    }
  });
  document.querySelectorAll('#d-routes .route').forEach(div => {
    const btn = div.querySelector('button');
    div.classList.toggle('active', btn && btn.dataset.route === routeName);
  });
}

function depsFor(id, route) {
  const n = NODE_BY_ID[id];
  if (!n) return [];
  const deps = new Set();
  GRAPH.edges.filter(e => e.from === id && e.kind === 'uses').forEach(e => deps.add(e.to));
  const routes = routesOf(id);
  if (routes.length) {
    let chosen = route != null ? routes.find(r => r.route === route) : null;
    if (!chosen) chosen = routes[0];
    chosen.deps.forEach(x => deps.add(x));
  }
  return [...deps];
}

function computeSupport(id, route) {
  const support = new Set();
  const stack = depsFor(id, route);
  while (stack.length) {
    const nid = stack.pop();
    if (support.has(nid) || !NODE_BY_ID[nid]) continue;
    support.add(nid);
    depsFor(nid, null).forEach(x => stack.push(x));
  }
  support.delete(id);
  return support;
}

function highlightSupport(id, route) {
  if (mode === 'overview') enterDetail();
  const support = computeSupport(id, route);
  const inSet = nid => nid === id || support.has(nid);
  cy.batch(() => {
    cy.nodes('.leaf').forEach(n => {
      const s = inSet(n.id());
      n.toggleClass('support', s);
      n.toggleClass('faded', !s);
    });
    cy.edges().removeClass('route-on');
    cy.edges().filter(e => !e.hasClass('cedge')).forEach(e => {
      e.toggleClass('faded', !(inSet(e.source().id()) && inSet(e.target().id())));
    });
  });
  const label = route === '_auto' ? '自動ルート' : `ルート ${route}`;
  document.getElementById('stats').textContent =
    `支持集合（${stripMath(NODE_BY_ID[id].title)} / ${label}）: ${support.size} ブロック`;
}

// ---------------------------------------------------------------------------
// TeX → 表示テキスト
// ---------------------------------------------------------------------------
const SYM = {
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε',
  zeta: 'ζ', eta: 'η', theta: 'θ', vartheta: 'ϑ', iota: 'ι', kappa: 'κ', lambda: 'λ',
  mu: 'μ', nu: 'ν', xi: 'ξ', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ', upsilon: 'υ',
  phi: 'φ', varphi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω',
  Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π', Sigma: 'Σ',
  Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω', nabla: '∇', partial: '∂', infty: '∞',
  to: '→', mapsto: '↦', times: '×', cdot: '·', leq: '≤', le: '≤', geq: '≥', ge: '≥',
  neq: '≠', in: '∈', notin: '∉', subset: '⊂', subseteq: '⊆', supset: '⊃', supseteq: '⊇',
  cup: '∪', cap: '∩', emptyset: '∅', forall: '∀', exists: '∃', Rightarrow: '⇒',
  rightarrow: '→', leftrightarrow: '↔', iff: '⇔', langle: '⟨', rangle: '⟩', sum: 'Σ',
  prod: '∏', int: '∫', sqrt: '√', pm: '±', approx: '≈', sim: '∼', equiv: '≡', circ: '∘',
  ldots: '…', dots: '…', cdots: '…', sharp: '#', setminus: '∖', otimes: '⊗',
  min: 'min', max: 'max', sup: 'sup', inf: 'inf', lim: 'lim', log: 'log', exp: 'exp', ln: 'ln',
  R: 'ℝ', N: 'ℕ', Z: 'ℤ', Q: 'ℚ', E: '𝔼', X: '𝒳', Y: '𝒴', Pp: '𝒫', Mm: 'ℳ',
  Bb: 'ℬ', Cc: '𝒞', Lcal: 'ℒ', Zz: '𝒵', Wass: '𝒲', WassD: 'W', MK: 'ℒ', MKD: 'L',
  Couplings: '𝒰', CouplingsD: 'Π', Potentials: 'ℛ', PotentialsD: 'R', ones: '1',
  Identity: 'I', simplex: 'Σ', defeq: ':=', dist: 'd', distD: 'D', CD: 'CD', Ric: 'Ric',
  KL: 'KL', KLD: 'KL', supp: 'supp', tr: 'tr', diag: 'diag',
  argmin: 'argmin', argmax: 'argmax', smin: 'smin', Tan: 'Tan', diverg: '∇·', d: 'd',
};

function mathToText(s) {
  if (!s) return '';
  let t = s;
  for (let i = 0; i < 5; i++) {
    t = t.replace(
      /\\(?:textbf|textit|textrm|textsf|emph|text|operatorname\*?|mathrm|mathbb|mathcal|mathbf|mathsf|mathfrak|boldsymbol|pmb|underline|mathop|hat|bar|tilde|vec|overline)\s*\{([^{}]*)\}/g,
      '$1');
  }
  t = t.replace(/\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, '$1/$2');
  t = t.replace(/\\sqrt\s*\{([^{}]*)\}/g, '√($1)');
  t = t.replace(/\\[A-Za-z]+/g, m => { const v = SYM[m.slice(1)]; return v !== undefined ? v : ''; });
  t = t.replace(/\\[\[\]()]/g, ' ')
       .replace(/\\[^A-Za-z]/g, ' ')
       .replace(/[\${}&]/g, '')
       .replace(/[\^_]/g, '')
       .replace(/~/g, ' ');
  return t.replace(/\s+/g, ' ').trim();
}

function texToHtml(s) {
  if (!s) return '<span class="muted">（本文なし）</span>';
  return s
    .replace(/\\textbf\{([^{}]*)\}/g, '<strong>$1</strong>')
    .replace(/\\(?:emph|textit)\{([^{}]*)\}/g, '<em>$1</em>')
    .replace(/\\textrm\{([^{}]*)\}/g, '$1');
}

function stripMath(s) {
  if (!s) return '';
  return s.replace(/\$([^$]*)\$/g, '$1')
          .replace(/\\[a-zA-Z]+/g, '').replace(/[{}]/g, '').trim();
}

load();
