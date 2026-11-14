import { defineConfig } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

// Browser tests of the built site. They run against web/out, served under the base path it was built for by
// scripts/serve-out.mjs, at 390 and 1280 px, on Chromium and on WebKit (the stand-in for Safari).
// Build first: npm run build:public (or let `npm run check` do it).
//
// Only Playwright's own bundled headless builds are used (npx playwright install chromium webkit). No system
// browser and no desktop application is started.

const OUT = path.join(__dirname, "out");
const INFO = path.join(OUT, "build-info.json");

interface BuildInfo {
  variant: "public" | "private";
  basePath: string;
  synthetic: boolean;
}

function buildInfo(): BuildInfo {
  if (!existsSync(INFO)) {
    throw new Error("web/out has no build-info.json. Build the site first: npm run build:public");
  }
  return JSON.parse(readFileSync(INFO, "utf8")) as BuildInfo;
}

const info = buildInfo();
const port = Number(process.env.E2E_PORT ?? 4173);
const origin = `http://127.0.0.1:${port}`;

const phone = { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 };
const laptop = { viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 };

export default defineConfig({
  testDir: "tests/e2e",
  testMatch: /.*\.spec\.ts$/,
  outputDir: "test-results/browser",
  reporter: [["list"]],
  forbidOnly: true,
  fullyParallel: true,
  workers: 3,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: `${origin}${info.basePath}/`,
    headless: true,
    trace: "off",
    screenshot: "off",
    video: "off",
  },
  webServer: {
    command: `node scripts/serve-out.mjs --dir out --base "${info.basePath}" --port ${port}`,
    url: `${origin}${info.basePath}/`,
    reuseExistingServer: false,
    timeout: 30_000,
  },
  projects: [
    { name: "chromium-390", use: { browserName: "chromium", ...phone } },
    { name: "chromium-1280", use: { browserName: "chromium", ...laptop } },
    { name: "webkit-390", use: { browserName: "webkit", ...phone } },
    { name: "webkit-1280", use: { browserName: "webkit", ...laptop } },
  ],
});
