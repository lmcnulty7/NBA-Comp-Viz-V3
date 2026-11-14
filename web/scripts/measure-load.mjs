// scripts/measure-load.mjs: first-view load time of a deployed site (PLAN.md C12).
//
// Opens each page in a fresh headless browser context (empty cache) with Playwright's bundled Chromium and
// WebKit, three times, and prints the median time to DOMContentLoaded, to the load event and to the largest
// contentful paint. It measures from the machine it runs on, over its own network: it says nothing about a
// phone on a mobile network.
//
//   node scripts/measure-load.mjs <site url> [page ...]

import { chromium, webkit } from "@playwright/test";

const [site, ...pages] = process.argv.slice(2);
if (!site) {
  console.error("usage: node scripts/measure-load.mjs <site url> [page ...]");
  process.exit(2);
}
const paths = pages.length ? pages : [""];
const RUNS = 3;

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

async function measure(browserType, url) {
  const browser = await browserType.launch();
  const results = [];
  for (let run = 0; run < RUNS; run += 1) {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.addInitScript(() => {
      window.__lcp = 0;
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) window.__lcp = Math.max(window.__lcp, entry.startTime);
      }).observe({ type: "largest-contentful-paint", buffered: true });
    });
    const response = await page.goto(url, { waitUntil: "load", timeout: 60000 });
    await page.waitForTimeout(500);
    const timing = await page.evaluate(() => {
      const nav = performance.getEntriesByType("navigation")[0];
      return { dcl: nav.domContentLoadedEventEnd, load: nav.loadEventEnd, lcp: window.__lcp || null,
        bytes: performance.getEntriesByType("resource").reduce((sum, r) => sum + (r.transferSize || 0), nav.transferSize || 0) };
    });
    results.push({ status: response.status(), ...timing });
    await context.close();
  }
  await browser.close();
  const pick = (key) => {
    const values = results.map((r) => r[key]).filter((v) => typeof v === "number");
    return values.length ? Math.round(median(values)) : null;
  };
  return { status: results[0].status, dcl: pick("dcl"), load: pick("load"), lcp: pick("lcp"), kb: Math.round(pick("bytes") / 1024) };
}

for (const path of paths) {
  const url = new URL(path, site.endsWith("/") ? site : `${site}/`).toString();
  for (const [name, type] of [["chromium", chromium], ["webkit", webkit]]) {
    const r = await measure(type, url);
    console.log(`${name.padEnd(9)} ${String(r.status).padEnd(4)} DOMContentLoaded ${r.dcl} ms, load ${r.load} ms, ` +
      `largest paint ${r.lcp ?? "n/a"} ms, ${r.kb} KB  ${url}`);
  }
}
