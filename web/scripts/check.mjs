// npm run check: every check of the web side, in order, stopping at the first failure.
//
//   types      the contract schema, the TypeScript types and the private patterns are current
//   typecheck  next typegen, then tsc --noEmit (TypeScript strict)
//   lint       eslint, with the rules that keep copy and data access where they belong
//   data       the data root validates against the contract (ajv, then the Python validator), no private string
//   copy       no digit, no dash and no banned phrase in web/content
//   reading    reading grade at most 9 and plain words on the cold open and the slides, copy equal to SCRIPT.md
//   contrast   WCAG AA contrast of the design tokens, light and dark
//   hygiene    no long dash and no private string in the web sources
//   unit       unit tests (geometry, scene, drawing, data source, formatting, the check scripts)
//   build      the public variant, under the GitHub Pages base path, from the public data root (PLAN.md D60)
//   frames     no SportVU frame data in any file of the public out/ (positions, clocks, ghosts, sensor dots)
//   methods    every number on the methods page is a number of methods.json, and every lane and test is there
//   video      no video file and no moving image in the public out/
//   browser    Playwright at 390 and 1280 px, on Chromium and WebKit, against out/ served under the base path
//
// Run it through the heavy-job lock, from the repository root:
//   python3 -m export.heavy --name "web check" -- npm --prefix web run check
// The browsers come from `npx playwright install chromium webkit` (Playwright's own headless builds).
import { spawnSync } from "node:child_process";
import path from "node:path";
import { PAGES_BASE_PATH, WEB_DIR, normaliseBasePath } from "./lib/paths.mjs";

const bin = (name) => path.join(WEB_DIR, "node_modules", ".bin", name);
const node = process.execPath;
const script = (name) => path.join(WEB_DIR, "scripts", name);

// The check builds for the GitHub Pages path unless the caller names another base path.
const basePath = normaliseBasePath(process.env.NEXT_PUBLIC_BASE_PATH ?? PAGES_BASE_PATH);
const env = { ...process.env, NEXT_PUBLIC_BASE_PATH: basePath, VARIANT: "public", NEXT_TELEMETRY_DISABLED: "1" };

const STEPS = [
  ["types", [[node, script("types.mjs"), "--check"]]],
  ["typecheck", [[bin("next"), "typegen"], [bin("tsc"), "--noEmit"]]],
  ["lint", [[bin("eslint"), "--max-warnings", "0"]]],
  ["data", [[node, script("validate-data.mjs"), "--python"]]],
  ["copy", [[node, script("check-copy.mjs")]]],
  ["reading", [[node, script("check-reading.mjs")]]],
  ["contrast", [[node, script("check-contrast.mjs")]]],
  ["hygiene", [[node, script("check-hygiene.mjs")]]],
  ["unit", [[bin("playwright"), "test", "--config", "playwright.unit.config.ts"]]],
  ["build", [[node, script("build.mjs"), "--variant", "public"]]],
  ["frames", [[node, script("check-frame-data.mjs")]]],
  ["methods", [[node, script("check-methods.mjs")]]],
  ["video", [[node, script("check-no-video.mjs")]]],
  ["browser", [[bin("playwright"), "test", "--config", "playwright.config.ts"]]],
];

const only = process.argv.slice(2).filter((name) => !name.startsWith("--"));
const unknown = only.filter((name) => !STEPS.some(([step]) => step === name));
if (unknown.length) {
  console.error(`check: unknown step(s) ${unknown.join(", ")}. Steps: ${STEPS.map(([step]) => step).join(", ")}`);
  process.exit(2);
}

const results = [];
let failed = null;
for (const [name, commands] of STEPS) {
  if (only.length && !only.includes(name)) continue;
  if (failed) {
    results.push([name, "not run", 0]);
    continue;
  }
  console.log(`\n[check] ${name}`);
  const started = Date.now();
  let status = 0;
  for (const [command, ...args] of commands) {
    const run = spawnSync(command, args, { cwd: WEB_DIR, stdio: "inherit", env });
    status = run.status ?? 1;
    if (run.error) console.error(`check: could not start ${path.basename(command)}: ${run.error.message}`);
    if (status !== 0) break;
  }
  const seconds = (Date.now() - started) / 1000;
  results.push([name, status === 0 ? "passed" : "FAILED", seconds]);
  if (status !== 0) failed = name;
}

console.log("\n[check] summary");
for (const [name, outcome, seconds] of results) {
  console.log(`  ${name.padEnd(10)} ${outcome.padEnd(8)} ${outcome === "not run" ? "" : `${seconds.toFixed(1)} s`}`);
}
if (failed) {
  console.error(`[check] failed at: ${failed}`);
  process.exit(1);
}
console.log(only.length ? `[check] passed (${only.join(", ")} only)` : "[check] passed");
