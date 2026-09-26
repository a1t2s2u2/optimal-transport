const blocksByName = new Map(
  (window.__blocks || []).map((block) => [block.name, block]),
);
const chapterFiles = window.__chapterFiles || {};
const typeLabels = window.__typeLabels || {};

function buildToc() {
  const toc = document.querySelector(".chapter-toc");
  const headings = [...document.querySelectorAll(".prose h2[id]")];
  if (!toc || headings.length === 0) return;

  toc.innerHTML = `<span class="chapter-toc__title">目次</span><ul class="chapter-toc__list">${headings
    .map(
      (heading) =>
        `<li><a class="chapter-toc__link" href="#${heading.id}">${heading.textContent}</a></li>`,
    )
    .join("")}</ul>`;

  const links = [...toc.querySelectorAll(".chapter-toc__link")];
  const observer = new IntersectionObserver(
    (entries) => {
      const current = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (!current) return;
      links.forEach((link) => {
        link.classList.toggle("is-active", link.hash === `#${current.target.id}`);
      });
    },
    { rootMargin: "-64px 0px -70%", threshold: 0 },
  );
  headings.forEach((heading) => observer.observe(heading));
}

function jumpHref(block) {
  if (
    block.chapter &&
    block.chapter !== window.__currentChapter &&
    chapterFiles[block.chapter]
  ) {
    return `${chapterFiles[block.chapter]}#${block.id}`;
  }
  return `#${block.id}`;
}

function cardHtml(block, closeButton = false) {
  const type = /^[a-z]+$/.test(block.type) ? block.type : "definition";
  const close = closeButton
    ? '<button class="ref-sidebar__close" type="button" aria-label="閉じる">&times;</button>'
    : "";
  return `<div class="ref-sidebar__card ref-sidebar__card--${type}">
    <div class="ref-sidebar__card-header">
      <span class="ref-sidebar__type">${typeLabels[block.type] || "参照"}</span>${close}
    </div>
    <div class="ref-sidebar__content">${block.html}</div>
    <a class="ref-sidebar__jump" href="${jumpHref(block)}">本文で見る &rarr;</a>
  </div>`;
}

function typeset(element) {
  if (window.MathJax?.typesetPromise) window.MathJax.typesetPromise([element]);
}

const sidebarBody = document.querySelector(".ref-sidebar__body");
const sheet = document.getElementById("ref-sheet");

function showReference(block) {
  if (window.matchMedia("(min-width: 1280px)").matches && sidebarBody) {
    sidebarBody.innerHTML = cardHtml(block, true);
    sidebarBody
      .querySelector(".ref-sidebar__close")
      ?.addEventListener("click", resetSidebar);
    typeset(sidebarBody);
    return;
  }

  const content = sheet?.querySelector(".ref-sheet__content");
  if (!sheet || !content) return;
  content.innerHTML = cardHtml(block);
  sheet.showModal();
  typeset(content);
}

function resetSidebar() {
  if (!sidebarBody) return;
  sidebarBody.innerHTML =
    '<p class="ref-sidebar__empty">参照リンクをクリックすると<br>ここに定義や定理が表示されます</p>';
}

document.addEventListener("click", (event) => {
  const ref = event.target.closest(".ref");
  if (!ref) return;
  const block = blocksByName.get(ref.dataset.ref);
  if (block) showReference(block);
});

sheet?.querySelector(".ref-sheet__close")?.addEventListener("click", () => sheet.close());
sheet?.addEventListener("click", (event) => {
  if (event.target === sheet) sheet.close();
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (sheet?.open) sheet.close();
  resetSidebar();
});

buildToc();
