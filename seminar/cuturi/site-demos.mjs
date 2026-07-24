// tex の \demohint{名前} から差し込む、このセミナー固有の図。
// tikz の図はサイトでは省略されるため、要点になるものだけ HTML で作り直す。

const DIAGRAM_STYLE = `
<style>
.tp-fig{position:relative;width:100%;max-width:460px;height:200px;margin:12px auto}
.tp-fig .node{position:absolute;width:76px;height:36px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;border:1.5px solid}
.tp-fig .node--f{background:#dbeafe;border-color:#3b82f6}
.tp-fig .node--s{background:#ffedd5;border-color:#f97316}
.tp-fig .lbl{position:absolute;font-size:13px;white-space:nowrap}
.tp-fig .edge-lbl{position:absolute;font-size:12px;background:rgba(255,255,255,.85);padding:0 3px;white-space:nowrap}
.tp-fig svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
</style>`;

// 工場とスーパーの間の輸送コスト網。
export function transportCostDiagram() {
  return `
${DIAGRAM_STYLE}
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

// 上の例に対する最適輸送計画。
export function transportOptimalDiagram() {
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
