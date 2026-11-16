// The parts of the polish score that need no browser (PLAN.md P4, docs/POLISH_LOOP.md): the weights, the
// scalar, the gate, the route sample, the first-load JavaScript of a page and the reading grade of the copy.
// scripts/polish-score.mjs does the measuring; this module only turns measurements into a score, so the unit
// tests can hold it to its rules.
//
// The scalar is a cost: lower is better. Each metric is divided by its value in the reference (the median of
// the baseline runs), so the reference scores 1.000, and the weights sum to one:
//
//   cost = sum over metrics of weight x (value + offset) / (reference + offset)
//
// The offset is 1 for a count that can be zero (the accessibility weight) and 0 elsewhere.
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { gzipSync } from "node:zlib";
import { fkGrade, wordsOf } from "./readability.mjs";

export const SCORE_SCHEMA = "polish-score/1";

// What each metric weighs in the cost, and why (docs/POLISH_LOOP.md, "Weights").
export const WEIGHTS = Object.freeze({
  a11yWeighted: 0.25, // accessibility: axe-core violations, weighted by impact, summed over pages and browsers
  firstViewMs: 0.25, // first view: largest paint (load event where a browser reports none), mean of page medians
  firstLoadJsKb: 0.2, // gzip size of the JavaScript a page loads before it is interactive, mean over pages
  pageWeightKb: 0.1, // bytes a first view fetches (images, fonts, data, scripts), mean of page medians
  readingGrade: 0.2, // Flesch-Kincaid grade of every sentence of the site copy, words weighted
});
export const OFFSETS = Object.freeze({ a11yWeighted: 1, firstViewMs: 0, firstLoadJsKb: 0, pageWeightKb: 0, readingGrade: 0 });
export const METRICS = Object.freeze(Object.keys(WEIGHTS));

// axe-core impact levels, weighted the way axe orders them.
export const IMPACT = Object.freeze({ minor: 1, moderate: 2, serious: 3, critical: 4 });

/** The cost of a set of metric values against a reference, and each metric's share of it. */
export function costOf(values, reference) {
  const terms = {};
  let cost = 0;
  for (const metric of METRICS) {
    const value = values[metric];
    const ref = reference[metric];
    if (typeof value !== "number" || typeof ref !== "number") throw new Error(`polish: no value for ${metric}`);
    const offset = OFFSETS[metric];
    if (ref + offset <= 0) throw new Error(`polish: the reference of ${metric} is ${ref}, so it cannot divide`);
    terms[metric] = round((value + offset) / (ref + offset), 4);
    cost += WEIGHTS[metric] * ((value + offset) / (ref + offset));
  }
  return { cost: round(cost, 4), terms };
}

export function round(value, digits = 1) {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

export function median(values) {
  const sorted = values.filter((value) => typeof value === "number").sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function mean(values) {
  const kept = values.filter((value) => typeof value === "number");
  return kept.length ? kept.reduce((sum, value) => sum + value, 0) / kept.length : null;
}

// ------------------------------------------------------------------------------------------------ the routes

/** Every page of a built site, as a path under the base path ("" is the home page). Data and clips are not pages. */
export function builtRoutes(outDir) {
  const routes = [];
  const walk = (dir, prefix) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (!entry.isDirectory() || entry.name.startsWith("_") || entry.name === "data" || entry.name === "clips") continue;
      const relative = `${prefix}${entry.name}/`;
      if (existsSync(path.join(dir, entry.name, "index.html"))) routes.push(relative);
      walk(path.join(dir, entry.name), relative);
    }
  };
  if (existsSync(path.join(outDir, "index.html"))) routes.push("");
  walk(outDir, "");
  return routes.sort();
}

// The family of a route: its first folder, then one star for each deeper folder (review, then star, star).
export function routePattern(route) {
  const parts = route.split("/").filter(Boolean);
  if (!parts.length) return "";
  return `${[parts[0], ...parts.slice(1).map(() => "*")].join("/")}/`;
}

function naturalCompare(a, b) {
  const left = a.split("/");
  const right = b.split("/");
  for (let i = 0; i < Math.max(left.length, right.length); i += 1) {
    const x = left[i] ?? "";
    const y = right[i] ?? "";
    if (x === y) continue;
    if (/^\d+$/.test(x) && /^\d+$/.test(y)) return Number(x) - Number(y);
    return x.localeCompare(y);
  }
  return 0;
}

/** One page per family, the first in natural order, so the sample is the same on every build of the same data. */
export function sampleRoutes(routes) {
  const first = new Map();
  for (const route of [...routes].sort(naturalCompare)) {
    const pattern = routePattern(route);
    if (!first.has(pattern)) first.set(pattern, route);
  }
  return [...first.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([pattern, route]) => ({ pattern, route }));
}

// ------------------------------------------------------------------------------------- first-load JavaScript

const SCRIPT_TAG = /<script\b[^>]*>/gi;
const STYLE_TAG = /<link\b[^>]*rel="stylesheet"[^>]*>/gi;

function attribute(tag, name) {
  const match = new RegExp(`\\s${name}="([^"]*)"`, "i").exec(tag);
  return match ? match[1] : null;
}

function localFile(outDir, basePath, href) {
  if (!href || /^[a-z]+:/i.test(href) || href.startsWith("//")) return null;
  let relative = href.split(/[?#]/)[0];
  if (basePath && relative.startsWith(`${basePath}/`)) relative = relative.slice(basePath.length);
  const file = path.join(outDir, relative);
  return existsSync(file) && statSync(file).isFile() ? file : null;
}

/**
 * The scripts and stylesheets a page's HTML asks for before it is interactive: every <script src> but the
 * nomodule fallback (a modern browser skips it), and every stylesheet link. Sizes are raw and gzip.
 */
export function firstLoadAssets(outDir, route, basePath) {
  const html = readFileSync(path.join(outDir, route, "index.html"), "utf8");
  const sum = (tags, hrefName, skip) => {
    const files = new Set();
    for (const tag of html.match(tags) ?? []) {
      if (skip(tag)) continue;
      const file = localFile(outDir, basePath, attribute(tag, hrefName));
      if (file) files.add(file);
    }
    let raw = 0;
    let gzip = 0;
    for (const file of files) {
      const bytes = readFileSync(file);
      raw += bytes.length;
      gzip += gzipSync(bytes).length;
    }
    return { files: files.size, raw, gzip };
  };
  return {
    js: sum(SCRIPT_TAG, "src", (tag) => /\snomodule/i.test(tag)),
    css: sum(STYLE_TAG, "href", () => false),
  };
}

// ----------------------------------------------------------------------------------------- the reading grade

function strings(value) {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.flatMap(strings);
  if (value && typeof value === "object") return Object.values(value).flatMap(strings);
  return [];
}

/** The on-screen lines of one copy file: every JSON string, or every Markdown line before "## Notes". */
function copyLines(file) {
  const text = readFileSync(file, "utf8");
  if (file.endsWith(".json")) return strings(JSON.parse(text));
  const lines = text.split("\n");
  const notesAt = lines.findIndex((line) => /^## Notes\s*$/.test(line));
  return (notesAt < 0 ? lines : lines.slice(0, notesAt))
    .filter((line) => !/^\s*(#|---|[a-z]+:\s)/.test(line))
    .map((line) => line.replace(/^\s*[-*]\s+/, "").trim());
}

function copyFiles(dir) {
  const files = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...copyFiles(full));
    else if (/\.(json|md)$/.test(entry.name)) files.push(full);
  }
  return files.sort();
}

/**
 * The reading grade of the site copy: the Flesch-Kincaid grade of each sentence (a line that ends in . ! or ?),
 * weighted by its words, over every copy file. Labels and headings are left out: one word is not a sentence.
 * Presenter notes are left out, as check-reading.mjs leaves them out.
 */
export function copyGrade(contentDir) {
  let words = 0;
  let weighted = 0;
  let sentences = 0;
  const perFile = {};
  for (const file of copyFiles(contentDir)) {
    let fileWords = 0;
    let fileWeighted = 0;
    for (const line of copyLines(file)) {
      if (!/[.!?]["')]?$/.test(line.trim())) continue;
      const count = wordsOf(line).length;
      if (!count) continue;
      const grade = fkGrade(line);
      fileWords += count;
      fileWeighted += grade * count;
      sentences += 1;
    }
    if (fileWords) perFile[path.relative(contentDir, file).split(path.sep).join("/")] = round(fileWeighted / fileWords, 2);
    words += fileWords;
    weighted += fileWeighted;
  }
  return { grade: words ? round(weighted / words, 3) : 0, words, sentences, perFile };
}

// ---------------------------------------------------------------------------------------------- the gate

/** The key of one accessibility finding: browser, width, page family and axe rule. */
export function a11yKey({ browser, width, pattern, rule }) {
  return `${browser}|${width}|${pattern || "home"}|${rule}`;
}

/**
 * The gate a candidate must pass whatever its cost (docs/POLISH_LOOP.md, "Gate"). Each reason it gives is one
 * way the candidate is rejected. `base` is the score of the site the candidate was made from; without it only
 * the candidate's own checks are applied.
 */
export function gateReasons(score, base) {
  const reasons = [];
  if (!score.check?.ran) reasons.push("npm run check was not run");
  else if (!score.check.passed) reasons.push(`npm run check failed at ${score.check.failedAt ?? "a step"}`);
  if (score.copy?.problems) reasons.push(`check-copy.mjs reports ${score.copy.problems} problem(s)`);
  if (score.reading?.problems) reasons.push(`check-reading.mjs reports ${score.reading.problems} problem(s)`);
  for (const failure of score.pageFailures ?? []) reasons.push(`page failure: ${failure}`);
  if (!base) return reasons;
  if (score.variant !== base.variant) reasons.push(`the variant is ${score.variant}, the base is ${base.variant}`);
  if (score.dataBuild !== base.dataBuild) {
    reasons.push(`the data build is ${score.dataBuild}, the base was scored on ${base.dataBuild}`);
  }
  if (score.routes?.hash !== base.routes?.hash) {
    reasons.push(`the built pages differ from the base (${score.routes?.count} pages, the base has ${base.routes?.count})`);
  }
  const before = base.a11y?.findings ?? {};
  for (const [key, found] of Object.entries(score.a11y?.findings ?? {})) {
    const had = before[key];
    if (!had) reasons.push(`new accessibility violation ${key} (${found.nodes} node(s))`);
    else if (found.nodes > had.nodes) reasons.push(`accessibility violation ${key} grew from ${had.nodes} to ${found.nodes} node(s)`);
  }
  return reasons;
}

/**
 * Keep or reject one candidate, given its scores and the base's per variant, and each variant's noise (the
 * spread of the cost over the baseline runs). Kept only when every gate passes, no variant's cost rises by more
 * than its noise, and at least one variant's cost falls by more than its noise.
 */
export function decide(variants) {
  const reasons = [];
  let improved = false;
  for (const [variant, { gate, baseCost, cost, noise }] of Object.entries(variants)) {
    for (const reason of gate) reasons.push(`${variant}: ${reason}`);
    const gain = baseCost - cost;
    if (gain < -noise) reasons.push(`${variant}: the cost rose from ${baseCost} to ${cost}, more than the noise ${noise}`);
    if (gain > noise) improved = true;
  }
  if (!reasons.length && !improved) reasons.push("no variant improved by more than its noise");
  return { keep: reasons.length === 0, reasons };
}
