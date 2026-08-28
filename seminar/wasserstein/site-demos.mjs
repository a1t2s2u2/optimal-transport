// Wasserstein 距離セミナーのサイト専用の概念図。

const GLUING_STYLE = `
<style>
.wg-gluing{position:relative;width:100%;max-width:640px;height:330px;margin:18px auto 8px}
.wg-gluing svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.wg-gluing .card{position:absolute;box-sizing:border-box;border:1.5px solid;border-radius:12px;padding:8px 12px;text-align:center;background:var(--paper,#fff);box-shadow:0 2px 5px rgba(15,23,42,.08);line-height:1.25}
.wg-gluing .symbol{display:block;font-weight:700;font-size:20px;line-height:1.1}
.wg-gluing .detail{display:block;font-size:12px;color:var(--ink-muted,#52606d);margin-top:3px}
.wg-gluing .pi{left:8%;top:8px;width:150px;border-color:#2563eb;background:#eff6ff}
.wg-gluing .sigma{right:8%;top:8px;width:150px;border-color:#ea580c;background:#fff7ed}
.wg-gluing .middle{left:50%;top:112px;transform:translateX(-50%);width:178px;border-color:#7c3aed;background:#f5f3ff}
.wg-gluing .x{left:12%;top:205px;width:164px;border-color:#2563eb;background:#eff6ff}
.wg-gluing .z{right:12%;top:205px;width:164px;border-color:#ea580c;background:#fff7ed}
.wg-gluing .gamma{left:50%;top:278px;transform:translateX(-50%);width:196px;border-color:#059669;background:#ecfdf5}
.wg-gluing .note{position:absolute;font-size:12px;color:var(--ink-muted,#52606d);font-weight:600;white-space:nowrap;background:var(--paper,#fff);padding:0 3px}
.wg-gluing .common{left:50%;top:87px;transform:translateX(-50%)}
.wg-gluing .conditional{left:50%;top:181px;transform:translateX(-50%)}
.wg-gluing .marginal{left:50%;top:256px;transform:translateX(-50%)}
@media (max-width:520px){.wg-gluing .pi{left:0}.wg-gluing .sigma{right:0}.wg-gluing .x{left:2%;width:148px}.wg-gluing .z{right:2%;width:148px}}
</style>`;

export function gluingDiagram() {
  return `
${GLUING_STYLE}
<figure aria-label="接着補題の構成" style="margin:1em 0">
  <div class="wg-gluing">
    <svg viewBox="0 0 640 330" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <marker id="wg-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#64748b"/></marker>
      </defs>
      <path d="M175 72 L282 126" fill="none" stroke="#64748b" stroke-width="1.7" marker-end="url(#wg-arrow)"/>
      <path d="M465 72 L358 126" fill="none" stroke="#64748b" stroke-width="1.7" marker-end="url(#wg-arrow)"/>
      <path d="M287 168 L184 219" fill="none" stroke="#64748b" stroke-width="1.7" marker-end="url(#wg-arrow)"/>
      <path d="M353 168 L456 219" fill="none" stroke="#64748b" stroke-width="1.7" marker-end="url(#wg-arrow)"/>
      <path d="M184 260 L283 294" fill="none" stroke="#64748b" stroke-width="1.7" marker-end="url(#wg-arrow)"/>
      <path d="M456 260 L357 294" fill="none" stroke="#64748b" stroke-width="1.7" marker-end="url(#wg-arrow)"/>
    </svg>
    <div class="card pi"><span class="symbol">π</span><span class="detail">(x, y) の分布</span></div>
    <div class="card sigma"><span class="symbol">σ</span><span class="detail">(y, z) の分布</span></div>
    <span class="note common">共通の周辺分布</span>
    <div class="card middle"><span class="symbol">y ∼ ν</span><span class="detail">一度だけ選んで共有する</span></div>
    <span class="note conditional">y が決まったもとで選ぶ</span>
    <div class="card x"><span class="symbol">π<sub>y</sub></span><span class="detail">y のもとでの x の分布</span></div>
    <div class="card z"><span class="symbol">σ<sub>y</sub></span><span class="detail">y のもとでの z の分布</span></div>
    <span class="note marginal">二つの周辺分布を保つ</span>
    <div class="card gamma"><span class="symbol">γ</span><span class="detail">(x, y, z) の分布</span></div>
  </div>
  <figcaption style="text-align:center;font-size:.9em;color:var(--ink-muted,#52606d)">
    共通の y のもとで π<sub>y</sub> と σ<sub>y</sub> を掛け合わせて γ を作る。
  </figcaption>
</figure>`;
}
