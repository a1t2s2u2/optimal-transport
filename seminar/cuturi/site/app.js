/* ==========================================================================
   Computational Optimal Transport — Interactions
   ========================================================================== */

/* ---------- Glossary ---------- */

const glossary = {
  polish: {
    title: "Polish 空間",
    body: "完備かつ可分な距離空間。確率測度の弱収束やカップリングの存在を扱いやすい。"
  },
  coupling: {
    title: "カップリング",
    body: "二つの周辺分布を固定した積空間上の確率測度。Kantorovich 緩和の未知量。"
  },
  entropy: {
    title: "離散エントロピー / エントロピー正則化",
    body: "離散エントロピーは \\(H(P)=-\\sum_{ij}P_{ij}(\\log P_{ij}-1)\\)。正則化では線形コストに \\(-\\varepsilon H(P)\\) を加える。"
  },
  kl: {
    title: "KL ダイバージェンス",
    body: "非負行列 \\(P,K\\) の差を測る量。\\(\\sum_{ij}P_{ij}\\log(P_{ij}/K_{ij})-P_{ij}+K_{ij}\\) で定義する。"
  },
  gibbs: {
    title: "Gibbs カーネル",
    body: "コスト行列から作る正行列。\\(K_{ij}=\\exp(-C_{ij}/\\varepsilon)\\)。正則化解は \\(P_\\varepsilon = \\mathrm{diag}(u) K \\mathrm{diag}(v)\\) の形をとる。"
  }
};

/* ---------- Reading Progress ---------- */

const progressFill = document.querySelector(".reading-progress__fill");

function updateProgress() {
  const scrollTop = window.scrollY;
  const docHeight = document.documentElement.scrollHeight - window.innerHeight;
  if (docHeight > 0 && progressFill) {
    progressFill.style.width = (scrollTop / docHeight) * 100 + "%";
  }
}

window.addEventListener("scroll", updateProgress, { passive: true });

/* ---------- Chapter TOC (Left Sidebar) ---------- */

function buildToc() {
  const toc = document.querySelector(".chapter-toc");
  if (!toc) return;

  const headings = document.querySelectorAll(".prose h2[id]");
  if (headings.length === 0) return;

  const title = document.createElement("span");
  title.className = "chapter-toc__title";
  title.textContent = "目次";
  toc.appendChild(title);

  const list = document.createElement("ul");
  list.className = "chapter-toc__list";

  headings.forEach((h) => {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = `#${h.id}`;
    a.className = "chapter-toc__link";
    a.textContent = h.textContent;
    li.appendChild(a);
    list.appendChild(li);
  });

  const progressBar = document.createElement("div");
  progressBar.className = "chapter-toc__progress";
  list.appendChild(progressBar);

  toc.appendChild(list);

  const updateTocProgress = () => {
    const active = list.querySelector(".chapter-toc__link.is-active");
    if (active && active.parentElement) {
      const li = active.parentElement;
      progressBar.style.top = li.offsetTop + "px";
      progressBar.style.height = li.offsetHeight + "px";
    }
  };

  const tocObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((e) => e.isIntersecting)
        .sort((a, b) => a.target.getBoundingClientRect().top - b.target.getBoundingClientRect().top)[0];
      if (!visible) return;
      list.querySelectorAll(".chapter-toc__link").forEach((link) => {
        link.classList.toggle(
          "is-active",
          link.getAttribute("href") === `#${visible.target.id}`
        );
      });
      updateTocProgress();
    },
    { rootMargin: "-60px 0px -75% 0px", threshold: 0 }
  );

  headings.forEach((h) => tocObserver.observe(h));
}

buildToc();

/* ---------- Reference Sidebar (Right, Fixed) ---------- */

const refSidebar = document.querySelector(".ref-sidebar");
const refBody = refSidebar?.querySelector(".ref-sidebar__body");
const refSheet = document.getElementById("ref-sheet");
const MAX_SIDEBAR_CARDS = 3;

const typeLabels = { definition: "定義", theorem: "定理", remark: "注意", example: "例", algorithm: "アルゴリズム" };
const typeColors = {
  definition: "var(--teal)",
  theorem: "var(--indigo)",
  remark: "var(--muted)",
  example: "var(--wine)",
  algorithm: "var(--amber)"
};

function blockJumpHref(block) {
  const current = window.__currentChapter;
  const files = window.__chapterFiles || {};
  if (block.chapter && block.chapter !== current && files[block.chapter]) {
    return `${files[block.chapter]}#${block.id}`;
  }
  return `#${block.id}`;
}

function showInSidebar(block) {
  if (!refBody) return;

  const existing = refBody.querySelector(`[data-block-id="${block.id}"]`);
  if (existing) return;

  const empty = refBody.querySelector(".ref-sidebar__empty");
  if (empty) empty.remove();

  const cards = refBody.querySelectorAll(".ref-sidebar__card");
  if (cards.length >= MAX_SIDEBAR_CARDS) {
    cards[cards.length - 1].remove();
  }

  const jumpHref = blockJumpHref(block);
  const card = document.createElement("div");
  card.className = "ref-sidebar__card";
  card.dataset.blockId = block.id;
  card.innerHTML =
    `<div class="ref-sidebar__card-header">` +
    `<span class="ref-sidebar__type" style="color:${typeColors[block.type] || "var(--teal)"}">${typeLabels[block.type] || "参照"}</span>` +
    `<button class="ref-sidebar__close" type="button" aria-label="閉じる">&times;</button>` +
    `</div>` +
    `<div class="ref-sidebar__content">${block.html}</div>` +
    `<a class="ref-sidebar__jump" href="${jumpHref}">本文で見る &rarr;</a>`;

  card.querySelector(".ref-sidebar__close").addEventListener("click", () => {
    card.remove();
    if (refBody.querySelectorAll(".ref-sidebar__card").length === 0) {
      refBody.innerHTML = '<p class="ref-sidebar__empty">参照リンクをクリックすると<br>ここに定義や定理が表示されます</p>';
    }
  });

  refBody.prepend(card);

  if (window.MathJax?.typesetPromise) {
    MathJax.typesetPromise([card]);
  }
}

function showRefMobile(block) {
  if (!refSheet) return;
  const content = refSheet.querySelector(".ref-sheet__content");
  if (!content) return;
  const label = typeLabels[block.type] || "参照";
  const jumpHref = blockJumpHref(block);
  content.innerHTML =
    `<h3 style="color:${typeColors[block.type] || "var(--teal)"}">${label}: ${block.name}</h3>` +
    `<div>${block.html}</div>` +
    `<p style="margin-top:12px"><a href="${jumpHref}" style="color:var(--teal);text-decoration:none">本文で見る &rarr;</a></p>`;
  refSheet.showModal();
  if (window.MathJax?.typesetPromise) {
    MathJax.typesetPromise([content]);
  }
}

function showBlock(block, sourceEl) {
  const hasWide = window.matchMedia("(min-width: 1400px)").matches;
  if (hasWide && refSidebar) {
    showInSidebar(block);
  } else {
    showRefMobile(block);
  }
}

/* ---------- Hover Tooltip ---------- */

let hoverTimeout = null;
let tooltip = null;

function showTooltip(refEl, block) {
  tooltip = document.createElement("div");
  tooltip.className = "ref-tooltip";
  const tmp = document.createElement("div");
  tmp.innerHTML = block.html;
  const text = tmp.textContent.slice(0, 100);
  tooltip.textContent = text + (tmp.textContent.length > 100 ? "..." : "");
  const rect = refEl.getBoundingClientRect();
  tooltip.style.top = rect.bottom + 6 + "px";
  tooltip.style.left = Math.min(rect.left, window.innerWidth - 340) + "px";
  document.body.appendChild(tooltip);
}

function hideTooltip() {
  clearTimeout(hoverTimeout);
  if (tooltip) {
    tooltip.remove();
    tooltip = null;
  }
}

document.addEventListener("mouseover", (e) => {
  const ref = e.target.closest(".ref");
  if (!ref) return;
  const blocks = window.__blocks || [];
  const block = blocks.find((b) => b.name === ref.dataset.ref);
  if (!block) return;
  hoverTimeout = setTimeout(() => showTooltip(ref, block), 400);
});

document.addEventListener("mouseout", (e) => {
  if (e.target.closest(".ref")) hideTooltip();
});

/* ---------- Click Handlers ---------- */

document.addEventListener("click", (e) => {
  const term = e.target.closest(".term");
  if (term) {
    const item = glossary[term.dataset.term];
    if (!item) return;
    const fakeBlock = {
      id: "glossary-" + term.dataset.term,
      name: item.title,
      type: "definition",
      chapter: window.__currentChapter,
      html: `<h3>${item.title}</h3><p>${item.body}</p>`
    };
    showBlock(fakeBlock, term);
    return;
  }

  const ref = e.target.closest(".ref");
  if (ref) {
    hideTooltip();
    const blocks = window.__blocks || [];
    const block = blocks.find((b) => b.name === ref.dataset.ref);
    if (!block) return;
    showBlock(block, ref);
    return;
  }
});

/* Close dialog */
if (refSheet) {
  const closeBtn = refSheet.querySelector(".ref-sheet__close");
  if (closeBtn) {
    closeBtn.addEventListener("click", () => refSheet.close());
  }
  refSheet.addEventListener("click", (e) => {
    if (e.target === refSheet) refSheet.close();
  });
}

/* ---------- Keyboard Navigation ---------- */

const allBlockEls = document.querySelectorAll(
  ".block, .example-band, .margin-note"
);
let currentBlockIdx = -1;

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

  if (e.key === "j") {
    currentBlockIdx = Math.min(currentBlockIdx + 1, allBlockEls.length - 1);
    allBlockEls[currentBlockIdx]?.scrollIntoView({ behavior: "smooth", block: "center" });
    e.preventDefault();
  }
  if (e.key === "k") {
    currentBlockIdx = Math.max(currentBlockIdx - 1, 0);
    allBlockEls[currentBlockIdx]?.scrollIntoView({ behavior: "smooth", block: "center" });
    e.preventDefault();
  }
  if (e.key === "Escape") {
    if (!kbdOverlay.hidden) {
      kbdOverlay.hidden = true;
      return;
    }
    if (refBody) {
      refBody.innerHTML = '<p class="ref-sidebar__empty">参照リンクをクリックすると<br>ここに定義や定理が表示されます</p>';
    }
    refSheet?.close();
  }
  if (e.key === "?") {
    kbdOverlay.hidden = !kbdOverlay.hidden;
    e.preventDefault();
  }
});

/* ---------- Block Scroll Fade-in ---------- */

const fadeObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add("is-visible");
        fadeObserver.unobserve(e.target);
      }
    });
  },
  { rootMargin: "0px 0px -60px 0px", threshold: 0.05 }
);

document.querySelectorAll(".block, .example-band, .margin-note").forEach((el) => {
  fadeObserver.observe(el);
});

/* ---------- Ref Click Pulse ---------- */

document.addEventListener("click", (e) => {
  const ref = e.target.closest(".ref");
  if (!ref) return;
  const blocks = window.__blocks || [];
  const block = blocks.find((b) => b.name === ref.dataset.ref);
  if (!block) return;
  const target = document.getElementById(block.id);
  if (target) {
    target.classList.remove("is-pulsing");
    void target.offsetWidth;
    target.classList.add("is-pulsing");
    setTimeout(() => target.classList.remove("is-pulsing"), 700);
  }
});

/* ---------- Keyboard Shortcut Overlay ---------- */

const kbdOverlay = document.createElement("div");
kbdOverlay.className = "kbd-overlay";
kbdOverlay.hidden = true;
kbdOverlay.innerHTML = `<div class="kbd-overlay__card">
  <h3>キーボードショートカット</h3>
  <div class="kbd-overlay__row"><span>次のブロックへ</span><span class="kbd-overlay__key">J</span></div>
  <div class="kbd-overlay__row"><span>前のブロックへ</span><span class="kbd-overlay__key">K</span></div>
  <div class="kbd-overlay__row"><span>サイドバーを閉じる</span><span class="kbd-overlay__key">Esc</span></div>
  <div class="kbd-overlay__row"><span>このヘルプ</span><span class="kbd-overlay__key">?</span></div>
</div>`;
document.body.appendChild(kbdOverlay);

kbdOverlay.addEventListener("click", (e) => {
  if (e.target === kbdOverlay) kbdOverlay.hidden = true;
});

/* ---------- Mermaid ---------- */

window.addEventListener("load", async () => {
  if (window.mermaid) {
    window.mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      themeVariables: {
        primaryColor: "#d9f0ed",
        primaryTextColor: "#20262d",
        primaryBorderColor: "#0f766e",
        lineColor: "#64707d",
        secondaryColor: "#f6e8c4",
        tertiaryColor: "#e1e4ff"
      }
    });
    if (window.mermaid.run) {
      await window.mermaid.run({ querySelector: ".mermaid" });
    } else if (window.mermaid.init) {
      window.mermaid.init(undefined, document.querySelectorAll(".mermaid"));
    }
  }
});
