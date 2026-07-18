// ヘッドレスで viewer を開き、描画・フィルタ・選択を検証する使い捨てスモークテスト。
const { chromium } = await import(process.env.PW_PATH || 'playwright');

const base = process.env.BASE || 'http://localhost:8765';
const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));

await page.goto(base + '/viewer/', { waitUntil: 'networkidle' });
await page.waitForTimeout(1500);

const errVisible = await page.locator('#error').isVisible();
const stats = await page.locator('#stats').textContent();
const nodeCount = await page.evaluate(() => cy ? cy.nodes().length : -1);
const edgeCount = await page.evaluate(() => cy ? cy.edges().length : -1);

// 空間フィルタ: metric を選ぶと polish 専用ノードが dim になるか
await page.selectOption('#space-filter', 'metric');
await page.waitForTimeout(300);
const dimmed = await page.evaluate(() => cy.nodes('.dim').map(n => n.id()));

// 双対定理を選択してルート A/B が出るか
const sel = await page.evaluate(() => {
  const ev = cy.getElementById('thm:sem-kantorovich-duality');
  return ev.length ? ev.id() : null;
});
await page.evaluate(() => { window.dispatchEvent(new Event('noop')); });
await page.evaluate(() => selectNode('thm:sem-kantorovich-duality'));
await page.waitForTimeout(300);
const routeCount = await page.locator('#d-routes .route').count();
const detailTitle = await page.locator('#d-title').textContent();

// 支持集合: ルート A を強調し、支持された(faded でない)ノード集合を取得
const supportA = await page.evaluate(() => {
  highlightSupport('thm:sem-kantorovich-duality', 'A');
  return cy.nodes('.support').map(n => n.id());
});

await page.screenshot({ path: 'viewer/_smoke.png', fullPage: false });
await browser.close();

console.log(JSON.stringify({
  errVisible, stats, nodeCount, edgeCount,
  dimmedCount: dimmed.length, dimmedSample: dimmed.slice(0, 5),
  selected: sel, routeCount, detailTitle,
  supportA, errors,
}, null, 2));

if (errVisible || nodeCount < 100 || routeCount < 2 || supportA.length < 1) {
  console.error('SMOKE FAIL');
  process.exit(1);
}
console.log('SMOKE OK');
