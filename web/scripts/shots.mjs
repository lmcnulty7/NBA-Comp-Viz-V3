// Screenshots of the built site, for looking at a page without opening a browser window (CLAUDE.md rule 11).
//
// It serves web/out on a free port, opens each page in Playwright's own headless Chromium or WebKit at 390 and
// 1280 px, in the light and the dark theme, and writes PNG files to test-results/shots (ignored by git).
// Run it through the heavy-job lock, after a build. It is not part of `npm run check`.
//
//   node scripts/shots.mjs [route ...]                     the default routes are the main pages
//   SHOT_ENGINES=chromium,webkit SHOT_SCHEMES=light,dark node scripts/shots.mjs review/
import { createRequire } from "node:module";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { BUILD_INFO_FILE, OUT_DIR, WEB_DIR, shown } from "./lib/paths.mjs";
import { createStaticServer } from "./serve-out.mjs";

const require = createRequire(import.meta.url);

const SIZES = [
  ["390", { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 }],
  ["1280", { viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 }],
];

function list(name, fallback) {
  return (process.env[name] ?? fallback).split(",").map((item) => item.trim()).filter(Boolean);
}

/** The main pages, with the first game, possession and player of the data build. */
function defaultRoutes() {
  const routes = ["", "review/", "prep/", "results/"];
  const read = (relative) => JSON.parse(readFileSync(path.join(OUT_DIR, "data", relative), "utf8"));
  const game = read("manifest.json").games[0];
  if (game) {
    routes.push(`review/${game.slug}/`);
    const row = read(`games/${game.slug}/possessions.json`).possessions[0];
    if (row) routes.push(`review/${game.slug}/${row.possession}/`);
  }
  const player = read("players.json").players[0];
  if (player) routes.push(`prep/${player.slug}/`);
  return routes;
}

async function main() {
  const infoFile = path.join(OUT_DIR, BUILD_INFO_FILE);
  if (!existsSync(infoFile)) {
    console.error("shots: no built site. Run npm run build:public or npm run build:private first.");
    return 1;
  }
  const info = JSON.parse(readFileSync(infoFile, "utf8"));
  const routes = process.argv.slice(2).length ? process.argv.slice(2) : defaultRoutes();
  const target = path.join(WEB_DIR, "test-results", "shots");
  mkdirSync(target, { recursive: true });
  const playwright = require("@playwright/test");
  const server = createStaticServer({ dir: OUT_DIR, base: info.basePath });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}${info.basePath}/`;
  let count = 0;
  try {
    for (const engine of list("SHOT_ENGINES", "chromium")) {
      const browser = await playwright[engine].launch({ headless: true });
      try {
        for (const [size, options] of SIZES) {
          for (const scheme of list("SHOT_SCHEMES", "light,dark")) {
            const context = await browser.newContext({ ...options, colorScheme: scheme });
            const page = await context.newPage();
            for (const route of routes) {
              await page.goto(base + route.replace(/^\/+/, ""), { waitUntil: "networkidle" });
              await page.evaluate(() => document.fonts.ready);
              const name = (route.replace(/^\/+|\/+$/g, "") || "home").replace(/[^a-zA-Z0-9-]+/g, "_");
              await page.screenshot({ path: path.join(target, `${name}-${size}-${scheme}-${engine}.png`), fullPage: true });
              count += 1;
            }
            await context.close();
          }
        }
      } finally {
        await browser.close();
      }
    }
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
  console.log(`shots: ${count} screenshot(s) in ${shown(target)}`);
  return 0;
}

process.exitCode = await main();
