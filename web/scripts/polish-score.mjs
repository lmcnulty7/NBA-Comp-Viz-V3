// The polish score of a built site (PLAN.md P4, docs/POLISH_LOOP.md): machine checks only, one JSON.
//
// It serves out/ on 127.0.0.1 under the base path it was built for (scripts/serve-out.mjs, with byte ranges for
// clips as a real host answers them) and measures, with
// Playwright's own bundled headless Chromium and WebKit (no system browser, no desktop application):
//   accessibility  axe-core (pinned dev dependency) on one page of each family, at 390 and 1280 px, in both
//                  browsers, WCAG 2.2 A and AA rules and axe's best practices, with reduced motion so an
//                  animation cannot change what is found;
//   first view     each sampled page loaded --runs times in a fresh context (empty cache) at 1280 px: time to
//                  DOMContentLoaded, to the load event and to the largest paint, and the bytes received before
//                  the load event; Chromium on an emulated phone connection and processor (CHROMIUM_THROTTLE);
//   bundle size    the gzip size of the JavaScript each sampled page's HTML loads (the nomodule fallback aside);
//   reading        the grade of the site copy (lib/polish.mjs copyGrade), and the problems of check-copy.mjs and
//                  check-reading.mjs, whose machine-written texts are read from out/data (staticChecks);
//   check          the outcome of npm run check for the variant, when --run-check runs it (it builds out/ first)
//                  or --check names the JSON a run of it wrote.
// The weights, the scalar (a cost, lower is better) and the gate are in scripts/lib/polish.mjs.
//
//   node scripts/polish-score.mjs [--run-check --variant public|private] [--check <check.json>]
//        [--repeat 3] [--runs 7] [--reference <baseline.json>] [--base <score.json>] [--json <file>] [--label <name>]
//
// --repeat N scores the same out/ N times; the reference is then the median of the runs and the noise is the
// spread of their costs. --reference takes the reference (and the noise) of a baseline instead. --base is the
// score of the site a candidate was made from, for the gate. It drives browsers and may build the site, so run
// it under the heavy-job lock, from the repository root:
//   python3 -m export.heavy --name "polish score" -- node web/scripts/polish-score.mjs --run-check --variant public
import { chromium, webkit } from "@playwright/test";
import axe from "axe-core";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { checkCopy } from "./check-copy.mjs";
import { checkReading } from "./check-reading.mjs";
import { BUILD_INFO_FILE, CONTENT_DIR, OUT_DIR, REPO_ROOT, normaliseBasePath } from "./lib/paths.mjs";
import { runCheck } from "./lib/polish-check.mjs";
import {
  IMPACT,
  METRICS,
  SCORE_SCHEMA,
  WEIGHTS,
  a11yKey,
  builtRoutes,
  copyGrade,
  costOf,
  firstLoadAssets,
  gateReasons,
  mean,
  median,
  round,
  sampleRoutes,
} from "./lib/polish.mjs";
import { createStaticServer } from "./serve-out.mjs";

const BROWSERS = [["chromium", chromium], ["webkit", webkit]];
const A11Y_WIDTHS = [390, 1280];
const TIMING_VIEWPORT = { width: 1280, height: 800 };
const AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];
const SETTLE_MS = 500;
// Chromium's first views are timed on an emulated mid-range phone connection and processor, so the time follows
// the bytes and the script a page needs rather than the speed of a local disk. WebKit offers no such emulation;
// it is timed as it is, in tens of milliseconds, where the run-to-run spread is near a sixth of the value, so it
// weighs less in the cost (lib/polish.mjs WEIGHTS).
const CHROMIUM_THROTTLE = { latencyMs: 40, downloadKbps: 10_000, uploadKbps: 5_000, cpuSlowdown: 4 };

function option(argv, name, fallback) {
  const index = argv.indexOf(name);
  return index >= 0 && argv[index + 1] !== undefined ? argv[index + 1] : fallback;
}

function readJson(file) {
  return JSON.parse(readFileSync(file, "utf8"));
}

function gitHead() {
  const head = spawnSync("git", ["rev-parse", "--short=12", "HEAD"], { cwd: REPO_ROOT, encoding: "utf8" });
  const dirty = spawnSync("git", ["status", "--porcelain", "--untracked-files=no"], { cwd: REPO_ROOT, encoding: "utf8" });
  return { commit: head.stdout.trim() || null, dirty: Boolean(dirty.stdout.trim()) };
}

// A navigation that has not reached its load event by then is tried again in a new context, up to TRIES times
// in all. A rare stall of a headless browser on a busy machine would otherwise end the whole score; the retries
// are counted in the score, and a page that stalls every time is a page failure.
const NAV_TIMEOUT_MS = 30_000;
const TRIES = 3;
const stalls = { count: 0 };

/** One page, opened in a fresh context; failures are uncaught errors, console errors and a bad status. */
async function openPage(browser, url, contextOptions, onPage = () => {}) {
  for (let attempt = 1; ; attempt += 1) {
    const context = await browser.newContext(contextOptions);
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (error) => errors.push(`uncaught ${error.message.split("\n")[0]}`));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(`console ${message.text().split("\n")[0]}`);
    });
    await onPage(page);
    try {
      const response = await page.goto(url, { waitUntil: "load", timeout: NAV_TIMEOUT_MS });
      return { context, page, errors, status: response?.status() ?? 0 };
    } catch (error) {
      await context.close();
      if (!/Timeout/.test(String(error?.message))) throw error;
      stalls.count += 1;
      if (attempt >= TRIES) return { context: null, page: null, errors: [`no load event after ${TRIES} tries`], status: 0 };
    }
  }
}

async function a11yPass(browserName, browser, origin, sample) {
  const findings = {};
  const failures = [];
  for (const width of A11Y_WIDTHS) {
    for (const { pattern, route } of sample) {
      const { context, page, errors, status } = await openPage(browser, `${origin}${route}`, {
        viewport: { width, height: 844 },
        reducedMotion: "reduce",
      });
      if (!page) {
        failures.push(`${browserName} ${width} /${route}: ${errors[0]}`);
        continue;
      }
      await page.waitForLoadState("networkidle", { timeout: 10_000 }).catch(() => {});
      await page.addScriptTag({ content: axe.source });
      const violations = await page.evaluate(
        async (tags) => (await window.axe.run(document, { runOnly: { type: "tag", values: tags }, resultTypes: ["violations"] }))
          .violations.map((v) => ({ rule: v.id, impact: v.impact, nodes: v.nodes.length })),
        AXE_TAGS,
      );
      for (const v of violations) {
        findings[a11yKey({ browser: browserName, width, pattern, rule: v.rule })] = { impact: v.impact ?? "minor", nodes: v.nodes };
      }
      if (status !== 200) failures.push(`${browserName} ${width} /${route}: status ${status}`);
      for (const error of errors) failures.push(`${browserName} ${width} /${route}: ${error}`);
      await context.close();
    }
  }
  return { findings, failures };
}

async function timingPass(browserName, browser, origin, sample, runs) {
  const rows = [];
  const failures = [];
  const load = async (route) => {
    let bytes = 0;
    const { context, page, errors, status } = await openPage(browser, `${origin}${route}`, { viewport: TIMING_VIEWPORT }, async (p) => {
      bytes = 0; // a retried navigation starts its count again
      // The bytes of the first view, before the load event; what the page fetches after it (route prefetches)
      // depends on timing and is left out. Chromium counts the body bytes it really received (its network log):
      // a clip asked for with an open range ("bytes=0-") is answered with the whole rest of the file, and the
      // browser stops reading once it has what it needs, so a declared length would count megabytes never sent.
      // WebKit has no such log; its count is the declared lengths, kept in the score for reading but out of the
      // page weight metric.
      let loaded = false;
      p.on("load", () => {
        loaded = true;
      });
      if (browserName === "chromium") {
        const cdp = await p.context().newCDPSession(p);
        cdp.on("Network.dataReceived", (event) => {
          if (!loaded) bytes += event.dataLength;
        });
        await cdp.send("Network.enable");
        await cdp.send("Network.emulateNetworkConditions", {
          offline: false,
          latency: CHROMIUM_THROTTLE.latencyMs,
          downloadThroughput: (CHROMIUM_THROTTLE.downloadKbps * 1000) / 8,
          uploadThroughput: (CHROMIUM_THROTTLE.uploadKbps * 1000) / 8,
        });
        await cdp.send("Emulation.setCPUThrottlingRate", { rate: CHROMIUM_THROTTLE.cpuSlowdown });
      }
      else {
        p.on("response", (response) => {
          if (!loaded) bytes += Number(response.headers()["content-length"] ?? 0);
        });
      }
      await p.addInitScript(() => {
        window.__lcp = 0;
        try {
          new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) window.__lcp = Math.max(window.__lcp, entry.startTime);
          }).observe({ type: "largest-contentful-paint", buffered: true });
        } catch {
          // a browser without largest-contentful-paint reports none
        }
      });
    });
    if (!page) return { status, errors, bytes, dcl: null, load: null, lcp: null };
    await page.waitForTimeout(SETTLE_MS);
    const timing = await page.evaluate(() => {
      const nav = performance.getEntriesByType("navigation")[0];
      return { dcl: nav.domContentLoadedEventEnd, load: nav.loadEventEnd, lcp: window.__lcp || null };
    });
    await context.close();
    return { status, errors, bytes, ...timing };
  };
  await load(sample[0].route); // warm the server and the browser once; not counted
  for (const { pattern, route } of sample) {
    const samples = [];
    for (let run = 0; run < runs; run += 1) samples.push(await load(route));
    for (const s of samples) {
      if (s.status !== 200) failures.push(`${browserName} timing /${route}: status ${s.status}`);
      for (const error of s.errors) failures.push(`${browserName} timing /${route}: ${error}`);
    }
    const pick = (key) => median(samples.map((s) => s[key]));
    const lcp = pick("lcp");
    rows.push({
      browser: browserName,
      pattern,
      route,
      dclMs: round(pick("dcl")),
      loadMs: round(pick("load")),
      lcpMs: lcp === null ? null : round(lcp),
      firstViewMs: round(median(samples.map((s) => s.lcp ?? s.load).filter((v) => typeof v === "number"))),
      kb: round(pick("bytes") / 1024),
      kbCounted: browserName === "chromium" ? "received" : "declared",
    });
  }
  return { rows, failures };
}

/** One measurement of the served site in both browsers. */
async function measureOnce(origin, sample, runs) {
  const findings = {};
  const timings = [];
  const failures = new Set();
  const versions = {};
  for (const [name, type] of BROWSERS) {
    const browser = await type.launch();
    versions[name] = browser.version();
    try {
      const a11y = await a11yPass(name, browser, origin, sample);
      Object.assign(findings, a11y.findings);
      a11y.failures.forEach((f) => failures.add(f));
      const timing = await timingPass(name, browser, origin, sample, runs);
      timings.push(...timing.rows);
      timing.failures.forEach((f) => failures.add(f));
    } finally {
      await browser.close();
    }
  }
  const a11yWeighted = Object.values(findings).reduce((sum, f) => sum + (IMPACT[f.impact] ?? 1) * f.nodes, 0);
  return {
    versions,
    findings,
    timings,
    failures: [...failures].map((f) => f.replaceAll(origin, "")).sort(),
    a11yWeighted,
    firstViewChromiumMs: round(mean(timings.filter((t) => t.browser === "chromium").map((t) => t.firstViewMs))),
    firstViewWebkitMs: round(mean(timings.filter((t) => t.browser === "webkit").map((t) => t.firstViewMs))),
    pageWeightKb: round(mean(timings.filter((t) => t.browser === "chromium").map((t) => t.kb))),
  };
}

/**
 * The static server of serve-out.mjs, plus byte ranges for a media file, as GitHub Pages and Cloudflare Pages
 * answer them. A browser asks for the start of a clip with a Range header to read its metadata; a server that
 * ignores the header sends the whole clip, and the first view would be charged for megabytes no real host sends.
 * Every other request goes to the serve-out handler unchanged.
 */
const RANGE_TYPES = { mp4: "video/mp4", webm: "video/webm" };
export function rangeAwareServer({ dir, base }) {
  const root = path.resolve(dir);
  const handler = createStaticServer({ dir, base }).listeners("request")[0];
  return createServer((request, response) => {
    const range = /^bytes=(\d*)-(\d*)$/.exec(request.headers.range ?? "");
    let pathname = "";
    try {
      pathname = decodeURIComponent(new URL(request.url ?? "/", "http://localhost").pathname);
    } catch {
      // a malformed path is the serve-out handler's to answer
    }
    const extension = path.extname(pathname).slice(1).toLowerCase();
    const inSite = !base || pathname.startsWith(`${base}/`);
    const file = path.join(root, pathname.slice(base.length));
    const servable =
      range && RANGE_TYPES[extension] && inSite && file.startsWith(root + path.sep) && existsSync(file) && statSync(file).isFile();
    if (!servable || (request.method !== "GET" && request.method !== "HEAD")) return handler(request, response);
    const size = statSync(file).size;
    const suffix = range[1] === "";
    const start = Math.max(0, suffix ? size - Number(range[2]) : Number(range[1]));
    const end = suffix || range[2] === "" ? size - 1 : Math.min(Number(range[2]), size - 1);
    if (start > end || start >= size) {
      response.writeHead(416, { "Content-Range": `bytes */${size}` });
      response.end();
      return;
    }
    response.writeHead(206, {
      "Content-Type": RANGE_TYPES[extension],
      "Content-Length": end - start + 1,
      "Content-Range": `bytes ${start}-${end}/${size}`,
      "Accept-Ranges": "bytes",
      "Cache-Control": "no-store",
    });
    if (request.method === "HEAD") response.end();
    else createReadStream(file, { start, end }).pipe(response);
  });
}

/**
 * The data root a built site was made from: its own data/ folder. The build copies the data root it read into
 * public/data and from there into out/data (for the public variant the public data root of D60, for the private
 * one the full root), so this is exactly the data the scored pages show, whatever SITE_DATA_DIR says now.
 */
export function builtDataDir(outDir) {
  return path.join(outDir, "data");
}

/**
 * The checks of the score that need no browser: check-copy.mjs on web/content, check-reading.mjs on web/content
 * and on the machine-written texts of the scored site's own data, and the grade of the copy.
 */
export function staticChecks(outDir) {
  return {
    copy: checkCopy(CONTENT_DIR),
    reading: checkReading({ contentDir: CONTENT_DIR, dataDir: builtDataDir(outDir) }),
    grade: copyGrade(CONTENT_DIR),
  };
}

/** Score out/ `repeat` times. Returns the score object (see docs/POLISH_LOOP.md, "The score file"). */
export async function score({ outDir = OUT_DIR, runs = 7, repeat = 1, check = null, reference = null, base = null, label = null } = {}) {
  const infoFile = path.join(outDir, BUILD_INFO_FILE);
  if (!existsSync(infoFile)) throw new Error("polish-score: out/ has no build-info.json; build the site first");
  const info = readJson(infoFile);
  const basePath = normaliseBasePath(info.basePath);

  const all = builtRoutes(outDir);
  const sample = sampleRoutes(all);
  const bundles = {};
  for (const { route } of sample) {
    const assets = firstLoadAssets(outDir, route, basePath);
    bundles[route || "home"] = {
      jsFiles: assets.js.files,
      jsRawKb: round(assets.js.raw / 1024),
      jsGzipKb: round(assets.js.gzip / 1024),
      cssGzipKb: round(assets.css.gzip / 1024),
    };
  }
  const firstLoadJsKb = round(mean(Object.values(bundles).map((b) => b.jsGzipKb)), 2);
  const { copy, reading, grade } = staticChecks(outDir);

  const server = rangeAwareServer({ dir: outDir, base: basePath });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const origin = `http://127.0.0.1:${server.address().port}${basePath}/`;
  const measured = [];
  try {
    for (let i = 0; i < repeat; i += 1) {
      console.error(`polish-score: measuring run ${i + 1} of ${repeat}`);
      measured.push(await measureOnce(origin, sample, runs));
    }
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }

  const valuesOf = (m) => ({
    a11yWeighted: m.a11yWeighted,
    firstViewChromiumMs: m.firstViewChromiumMs,
    firstViewWebkitMs: m.firstViewWebkitMs,
    firstLoadJsKb,
    pageWeightKb: m.pageWeightKb,
    readingGrade: grade.grade,
  });
  const runValues = measured.map(valuesOf);
  const values = Object.fromEntries(METRICS.map((metric) => [metric, median(runValues.map((v) => v[metric]))]));
  let ref;
  let noise = null;
  if (reference) {
    ref = { values: reference.reference.values, from: reference.label ?? "reference file" };
    noise = reference.noise ?? null;
  } else {
    ref = { values, from: repeat > 1 ? `median of ${repeat} runs` : "this run" };
  }
  const runRows = measured.map((m, i) => ({
    values: runValues[i],
    ...costOf(runValues[i], ref.values),
    failures: m.failures,
  }));
  if (!reference && repeat > 1) {
    const costs = runRows.map((r) => r.cost);
    noise = {
      runs: repeat,
      costs,
      costSpread: round(Math.max(...costs) - Math.min(...costs), 4),
      perMetric: Object.fromEntries(
        METRICS.map((metric) => {
          const xs = runValues.map((v) => v[metric]);
          return [metric, { min: Math.min(...xs), max: Math.max(...xs) }];
        }),
      ),
    };
  }

  // The findings of a repeated score are the most nodes any run found for each key, so a base is not lenient
  // about a violation that only some runs see.
  const findings = {};
  for (const m of measured) {
    for (const [key, f] of Object.entries(m.findings)) {
      if (!findings[key] || f.nodes > findings[key].nodes) findings[key] = f;
    }
  }
  const pageFailures = [...new Set(measured.flatMap((m) => m.failures))].sort();
  const result = {
    schema: SCORE_SCHEMA,
    label,
    variant: info.variant,
    basePath,
    synthetic: Boolean(info.synthetic),
    dataBuild: info.dataBuild ?? null,
    ...gitHead(),
    measuredWith: {
      browsers: measured[0].versions,
      axe: axe.version,
      axeTags: AXE_TAGS,
      a11yWidths: A11Y_WIDTHS,
      timingViewport: TIMING_VIEWPORT,
      chromiumThrottle: CHROMIUM_THROTTLE,
      loadsPerPage: runs,
      repeat,
    },
    weights: WEIGHTS,
    check: check ?? { ran: false },
    copy: { problems: copy.problems.length, files: copy.files },
    reading: { problems: reading.problems.length, generatedTexts: reading.generated, copyGrade: grade, script: reading.report },
    routes: { count: all.length, hash: createHash("sha256").update(all.join("\n")).digest("hex").slice(0, 16), sampled: sample },
    bundles,
    a11y: {
      weighted: values.a11yWeighted,
      violations: Object.keys(findings).length,
      nodes: Object.values(findings).reduce((sum, f) => sum + f.nodes, 0),
      findings: Object.fromEntries(Object.entries(findings).sort(([a], [b]) => a.localeCompare(b))),
    },
    timings: measured.map((m) => m.timings),
    pageFailures,
    navigationRetries: stalls.count,
    values,
    reference: ref,
    ...costOf(values, ref.values),
    runs: runRows,
    noise,
  };
  const reasons = gateReasons(result, base);
  result.gate = { passed: reasons.length === 0, against: base ? base.label ?? base.commit ?? "base" : null, reasons };
  return result;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const argv = process.argv.slice(2);
  if (!process.env.V3_HEAVY_HELD) {
    console.error(
      "polish-score: this drives browsers (and with --run-check builds the site), so it runs under the heavy-job lock:\n" +
        '  python3 -m export.heavy --name "polish score" -- node web/scripts/polish-score.mjs ...',
    );
    process.exit(2);
  }
  const outDir = path.resolve(option(argv, "--out", OUT_DIR));
  let check = option(argv, "--check") ? readJson(path.resolve(option(argv, "--check"))) : null;
  if (argv.includes("--run-check")) {
    const variant = option(argv, "--variant");
    if (!variant) {
      console.error("polish-score: --run-check needs --variant public or --variant private");
      process.exit(2);
    }
    check = runCheck(variant);
  }
  const reference = option(argv, "--reference") ? readJson(path.resolve(option(argv, "--reference"))) : null;
  const base = option(argv, "--base") ? readJson(path.resolve(option(argv, "--base"))) : null;
  try {
    const result = await score({
      outDir,
      runs: Number(option(argv, "--runs", "7")),
      repeat: Number(option(argv, "--repeat", "1")),
      check,
      reference,
      base,
      label: option(argv, "--label", null),
    });
    const text = `${JSON.stringify(result, null, 2)}\n`;
    const target = option(argv, "--json");
    if (target) {
      mkdirSync(path.dirname(path.resolve(target)), { recursive: true });
      writeFileSync(path.resolve(target), text);
    } else {
      process.stdout.write(text);
    }
    const noise = result.noise ? `, noise ${result.noise.costSpread ?? "from reference"}` : "";
    console.error(
      `polish-score: ${result.variant}, cost ${result.cost} (${result.reference.from}${noise}); ` +
        METRICS.map((m) => `${m} ${result.values[m]}`).join(", ") +
        `; gate ${result.gate.passed ? "passed" : `REJECTED: ${result.gate.reasons.join("; ")}`}`,
    );
    process.exitCode = result.gate.passed ? 0 : 1;
  } catch (error) {
    console.error(`polish-score: ${error.stack ?? error.message}`);
    process.exitCode = 2;
  }
}
