// The polish loop driver (PLAN.md P4, D76; docs/POLISH_LOOP.md). It keeps a change to the site only when the
// machine checks say the site got better by more than the noise of the measurement.
//
//   baseline   score the checked-out site --repeat times per variant, after npm run check, and write
//              baseline-<variant>.json (the reference for the cost and the noise) into --dir
//   start      make the loop branch polish/<run>/loop from --base and write its baseline into polish-runs/<run>
//   candidate  one proposed change, as a patch: a branch polish/<run>/cNNN from the loop branch, the patch applied
//              and refused if it touches a protected path (lib/polish-guard.mjs), committed, checked and scored per
//              variant against the best site so far, kept (the loop branch moves to it) or rejected, and logged
//   status     the log of a run, one line per candidate
//
//   node scripts/polish-loop.mjs baseline [--variants public,private] [--repeat 3] [--runs 7] [--dir reports/polish]
//   node scripts/polish-loop.mjs start --run <id> [--base <git ref>] [--variants public,private] [--repeat 3]
//        [--baseline <dir>]   reuse baseline-<variant>.json from <dir> when the site and the scorer are unchanged since
//   node scripts/polish-loop.mjs candidate --run <id> --patch <file> --message "<what the change does>"
//   node scripts/polish-loop.mjs status --run <id>
//
// Every command but status builds the site and drives browsers, so it runs under the heavy-job lock, from the
// repository root, one command per lock:
//   python3 -m export.heavy --name "polish candidate" -- node web/scripts/polish-loop.mjs candidate ...
// It never pushes, never merges into another branch than its own loop branch, and never edits a check, a test,
// the scorer or itself: the guard refuses such a patch before anything is built, and the guard test in the
// unit step of npm run check (POLISH_BASE) refuses it again.
import { spawnSync } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { REPO_ROOT } from "./lib/paths.mjs";
import { runCheck } from "./lib/polish-check.mjs";
import { forbiddenChanges } from "./lib/polish-guard.mjs";
import { decide } from "./lib/polish.mjs";
import { score } from "./polish-score.mjs";

// Three runs can understate the spread of the cost, so the noise a candidate must beat is never below this.
export const MIN_NOISE = 0.005;
// Outside web/, so the hygiene check of the site sources never sees it, and ignored by git (.gitignore).
export const RUNS_DIR = path.join(REPO_ROOT, "polish-runs");
const VARIANTS = ["public", "private"];

function option(argv, name, fallback) {
  const index = argv.indexOf(name);
  return index >= 0 && argv[index + 1] !== undefined ? argv[index + 1] : fallback;
}

function fail(message) {
  console.error(`polish-loop: ${message}`);
  process.exit(1);
}

function git(args, { allowFail = false } = {}) {
  const run = spawnSync("git", args, { cwd: REPO_ROOT, encoding: "utf8" });
  if (run.status !== 0 && !allowFail) fail(`git ${args.join(" ")} failed: ${(run.stderr || run.stdout).trim()}`);
  return { status: run.status ?? 1, out: (run.stdout ?? "").trim(), err: (run.stderr ?? "").trim() };
}

function requireLock() {
  if (!process.env.V3_HEAVY_HELD) {
    fail(
      "this command builds the site and drives browsers, so it runs under the heavy-job lock, from the repository root:\n" +
        '  python3 -m export.heavy --name "polish loop" -- node web/scripts/polish-loop.mjs ...',
    );
  }
}

function requireClean() {
  const dirty = git(["status", "--porcelain", "--untracked-files=all"]).out;
  if (dirty) fail(`the working tree is not clean; commit or stash first:\n${dirty}`);
}

function variantsOf(argv) {
  const list = option(argv, "--variants", VARIANTS.join(",")).split(",").map((v) => v.trim()).filter(Boolean);
  for (const variant of list) if (!VARIANTS.includes(variant)) fail(`unknown variant ${variant}`);
  return list;
}

function readJson(file) {
  return JSON.parse(readFileSync(file, "utf8"));
}

function writeJson(file, value) {
  mkdirSync(path.dirname(file), { recursive: true });
  writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}

/** A path as it is written into a log or a state file: relative to the repository, never absolute. */
function relative(file) {
  return path.relative(REPO_ROOT, file).split(path.sep).join("/");
}

function runDir(run) {
  if (!/^[a-z0-9][a-z0-9-]{0,40}$/.test(run ?? "")) fail("--run is a short name of lower-case letters, digits and hyphens");
  return path.join(RUNS_DIR, run);
}

/** Check, then score the checked-out site `repeat` times per variant; write baseline-<variant>.json into dir. */
async function baseline(variants, { repeat, runs, dir, label }) {
  const written = {};
  for (const variant of variants) {
    const check = runCheck(variant);
    if (!check.passed) fail(`npm run check failed for the ${variant} variant at ${check.failedAt}; no baseline is written`);
    const result = await score({ runs, repeat, check, label: `${label} ${variant}` });
    if (!result.gate.passed) fail(`the ${variant} baseline fails its own gate: ${result.gate.reasons.join("; ")}`);
    const file = path.join(dir, `baseline-${variant}.json`);
    writeJson(file, result);
    written[variant] = { file: relative(file), cost: result.cost, noise: result.noise.costSpread };
    console.log(`polish-loop: ${variant} baseline: cost ${result.cost}, noise ${result.noise.costSpread}, ${relative(file)}`);
  }
  return written;
}

/**
 * A stored baseline, copied into the run: allowed only when nothing under web/ changed between the commit it
 * was scored on and the base of the run, so it measured the same site.
 */
function reuseBaseline(variants, from, baseCommit, dir) {
  const written = {};
  for (const variant of variants) {
    const source = path.join(from, `baseline-${variant}.json`);
    if (!existsSync(source)) fail(`no stored baseline ${relative(source)}`);
    const stored = readJson(source);
    if (!stored.commit || stored.dirty) fail(`${relative(source)} was not scored on a clean commit`);
    // Everything under web/ builds or measures the site, except the driver itself and the tests.
    const changed = git(["diff", "--name-only", stored.commit, baseCommit, "--", "web",
      ":(exclude)web/scripts/polish-loop.mjs", ":(exclude)web/tests"]).out;
    if (changed) fail(`the site or the scorer changed since ${relative(source)} was scored (${changed.split("\n").length} file(s) under web/); start without --baseline`);
    const file = path.join(dir, `baseline-${variant}.json`);
    writeJson(file, stored);
    written[variant] = { file: relative(file), cost: stored.cost, noise: stored.noise.costSpread };
  }
  return written;
}

function log(dir, entry) {
  appendFileSync(path.join(dir, "log.jsonl"), `${JSON.stringify({ at: new Date().toISOString(), ...entry })}\n`);
}

async function start(argv) {
  requireLock();
  requireClean();
  const run = option(argv, "--run");
  const dir = runDir(run);
  if (existsSync(path.join(dir, "state.json"))) fail(`run ${run} exists already`);
  // The loop branch and its candidates are siblings (polish/<run>/loop, polish/<run>/c001): git cannot hold a
  // branch polish/<run> beside branches under polish/<run>/.
  const branch = `polish/${run}/loop`;
  const base = option(argv, "--base", "HEAD");
  const baseCommit = git(["rev-parse", base]).out;
  git(["switch", "-c", branch, baseCommit]);
  const variants = variantsOf(argv);
  const repeat = Number(option(argv, "--repeat", "3"));
  const runs = Number(option(argv, "--runs", "7"));
  const stored = option(argv, "--baseline");
  const written = stored
    ? reuseBaseline(variants, path.resolve(REPO_ROOT, stored), baseCommit, dir)
    : await baseline(variants, { repeat, runs, dir, label: `polish ${run} baseline` });
  const state = {
    run,
    branch,
    base,
    baseCommit,
    variants,
    runs,
    next: 1,
    // The best site so far: at the start, the baseline itself.
    best: Object.fromEntries(variants.map((v) => [v, { commit: baseCommit, file: written[v].file, cost: written[v].cost }])),
    noise: Object.fromEntries(variants.map((v) => [v, Math.max(written[v].noise, MIN_NOISE)])),
    reference: Object.fromEntries(variants.map((v) => [v, written[v].file])),
  };
  writeJson(path.join(dir, "state.json"), state);
  log(dir, { event: "start", branch, baseCommit, variants, baseline: written, noise: state.noise });
  console.log(`polish-loop: run ${run} started on ${branch} at ${baseCommit.slice(0, 12)}; noise ${JSON.stringify(state.noise)}`);
}

async function candidate(argv) {
  requireLock();
  requireClean();
  const run = option(argv, "--run");
  const dir = runDir(run);
  const stateFile = path.join(dir, "state.json");
  if (!existsSync(stateFile)) fail(`no run ${run}; start it first`);
  const state = readJson(stateFile);
  const patch = option(argv, "--patch");
  const message = option(argv, "--message", "").trim();
  if (!patch || !existsSync(patch)) fail("--patch names the file of the proposed change (a git diff)");
  if (!message) fail("--message says what the change does");

  const id = `c${String(state.next).padStart(3, "0")}`;
  const branch = `polish/${state.run}/${id}`;
  state.next += 1;
  writeJson(stateFile, state);
  const entry = { event: "candidate", id, branch, message, patch: path.basename(patch) };
  const reject = (reason, extra = {}) => {
    log(dir, { ...entry, ...extra, decision: "rejected", reasons: [reason].flat() });
    console.log(`polish-loop: ${id} rejected: ${[reason].flat().join("; ")}`);
  };

  git(["switch", state.branch]);
  git(["switch", "-c", branch]);
  const applied = git(["apply", "--index", "--whitespace=nowarn", path.resolve(patch)], { allowFail: true });
  const files = git(["diff", "--cached", "--name-only", "--no-renames"]).out.split("\n").filter(Boolean);
  const problems = applied.status === 0 ? forbiddenChanges(files) : [];
  if (applied.status !== 0 || files.length === 0 || problems.length) {
    // Nothing was committed: the patch is taken back out and the empty candidate branch removed.
    git(["reset", "-q", "--hard", "HEAD"]);
    git(["switch", state.branch]);
    git(["branch", "-D", branch]);
    if (applied.status !== 0) return reject(`the patch does not apply: ${applied.err.split("\n")[0]}`);
    if (!files.length) return reject("the patch changes nothing");
    return reject(problems.map((p) => `guard: ${p}`), { files });
  }
  git(["commit", "-q", "-m", `polish ${state.run} ${id}: ${message}`]);
  const commit = git(["rev-parse", "HEAD"]).out;

  const results = {};
  for (const variant of state.variants) {
    const check = runCheck(variant, { POLISH_BASE: state.branch });
    if (!check.passed) {
      results[variant] = { gate: [`npm run check failed at ${check.failedAt}`], baseCost: state.best[variant].cost, cost: null, noise: state.noise[variant] };
      break;
    }
    const scored = await score({
      runs: state.runs,
      check,
      reference: readJson(path.join(REPO_ROOT, state.reference[variant])),
      base: readJson(path.join(REPO_ROOT, state.best[variant].file)),
      label: `polish ${state.run} ${id} ${variant}`,
    });
    const file = path.join(dir, `${id}-${variant}.json`);
    writeJson(file, scored);
    results[variant] = { gate: scored.gate.reasons, baseCost: state.best[variant].cost, cost: scored.cost, noise: state.noise[variant], file: relative(file) };
  }
  // A variant whose check failed has no cost: the candidate is rejected on its gate reasons alone.
  const incomplete = state.variants.some((v) => !results[v] || results[v].cost === null);
  const verdict = incomplete
    ? { keep: false, reasons: Object.entries(results).flatMap(([v, r]) => r.gate.map((g) => `${v}: ${g}`)) }
    : decide(results);

  git(["switch", state.branch]);
  if (verdict.keep) {
    git(["merge", "-q", "--ff-only", branch]);
    for (const variant of state.variants) state.best[variant] = { commit, file: results[variant].file, cost: results[variant].cost };
    writeJson(stateFile, state);
  }
  log(dir, { ...entry, commit, files, variants: results, decision: verdict.keep ? "kept" : "rejected", reasons: verdict.reasons });
  console.log(`polish-loop: ${id} ${verdict.keep ? "kept" : `rejected: ${verdict.reasons.join("; ")}`}`);
}

function status(argv) {
  const dir = runDir(option(argv, "--run"));
  const file = path.join(dir, "log.jsonl");
  if (!existsSync(file)) fail("no log for this run");
  for (const line of readFileSync(file, "utf8").split("\n").filter(Boolean)) {
    const e = JSON.parse(line);
    if (e.event === "start") {
      console.log(`start ${e.baseCommit.slice(0, 12)} noise ${JSON.stringify(e.noise)}`);
      continue;
    }
    const costs = Object.entries(e.variants ?? {}).map(([v, r]) => `${v} ${r.baseCost} -> ${r.cost ?? "n/a"}`).join(", ");
    console.log(`${e.id} ${e.decision.padEnd(8)} ${costs}  ${e.message}${e.reasons?.length ? `  (${e.reasons.join("; ")})` : ""}`);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [command, ...argv] = process.argv.slice(2);
  try {
    if (command === "baseline") {
      requireLock();
      await baseline(variantsOf(argv), {
        repeat: Number(option(argv, "--repeat", "3")),
        runs: Number(option(argv, "--runs", "7")),
        dir: path.resolve(REPO_ROOT, option(argv, "--dir", "reports/polish")),
        label: option(argv, "--label", "baseline"),
      });
    } else if (command === "start") await start(argv);
    else if (command === "candidate") await candidate(argv);
    else if (command === "status") status(argv);
    else fail("usage: polish-loop.mjs baseline | start | candidate | status (see the head of this file)");
  } catch (error) {
    fail(error.stack ?? error.message);
  }
}
