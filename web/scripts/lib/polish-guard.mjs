// What a polish candidate may change (PLAN.md P4: the check scripts are read-only to the loop).
//
// A candidate changes the site: its pages, components, client code, copy, styles and static files. It never
// changes what judges the site: the check scripts, the scorer and the loop driver (all of web/scripts), the
// tests, the copy rules (web/scripts/lib/readability.mjs, the eslint copy rules, docs/SCRIPT.md), the data
// contract, the build and test configuration, the dependencies, the data and anything outside the web site.
// A path is allowed only when it is under an allowed folder and under no protected one, so a new kind of file
// is refused until someone decides it belongs to the site.
//
// web/lib holds checks as well as presentation, so only its named presentation modules are open to a candidate.
// The rest judge or carry the numbers: the page's number check of machine-written texts (genai.ts), the
// formatters export/genai.py mirrors (format.ts), the token table of docs/SCRIPT.md (tokens.ts), the one path
// from copy to a page (copy.ts), the DataSource and its schema version check (data/), the variant and base path
// the public build relies on (site.ts), the slide reader the reading check shares (talk.ts) and the source links
// of numbers (repo.ts). A new file under web/lib is refused too: the lint rule that keeps numbers and text out of
// code covers web/app and web/components only, so a new module there could carry a typed number.
//
// A candidate may change the pages that show machine-written texts, but it may not drop a call of the page's
// number check from them: the count of each call in REQUIRED_CALLS over web/app and web/components may not fall.
//
//   node scripts/lib/polish-guard.mjs --base <git ref>    exit 1 when HEAD changes a protected path since <ref>
import { spawnSync } from "node:child_process";
import { REPO_ROOT } from "./paths.mjs";

export const ALLOWED = Object.freeze([
  "web/app/",
  "web/components/",
  "web/lib/nav.ts",
  "web/lib/media.ts",
  "web/content/",
  "web/public/",
]);

// Each protected path, with what it is, so a refusal says why.
export const PROTECTED = Object.freeze([
  ["web/scripts/", "the check scripts, the scorer, the loop driver and the copy rules"],
  ["web/tests/", "the web tests"],
  ["tests/", "the Python tests"],
  ["web/lib/types/", "the data contract"],
  ["web/lib/genai.ts", "the page's number check of machine-written texts"],
  ["web/lib/format.ts", "the formatters the number check and export/genai.py mirror"],
  ["web/lib/tokens.ts", "the token table of docs/SCRIPT.md"],
  ["web/lib/copy.ts", "the one path from copy to a page"],
  ["web/lib/data/", "the DataSource and its schema version check"],
  ["web/lib/site.ts", "the variant and base path the public build relies on"],
  ["web/lib/talk.ts", "the slide reader the reading check shares"],
  ["web/lib/repo.ts", "the source links of numbers"],
  ["web/public/data/", "the site data"],
  ["web/eslint.config.mjs", "the lint and copy rules"],
  ["web/playwright.config.ts", "the browser test configuration"],
  ["web/playwright.unit.config.ts", "the unit test configuration"],
  ["web/package.json", "the dependencies"],
  ["web/package-lock.json", "the dependencies"],
  ["web/tsconfig.json", "the TypeScript configuration"],
  ["web/next.config.ts", "the build configuration"],
  ["docs/SCRIPT.md", "the approved copy"],
  ["docs/POLISH_LOOP.md", "the loop's rules"],
  ["reports/", "the reports and the polish baseline"],
  ["export/", "the Python export layer"],
  ["tools/", "the frozen-model tools"],
  ["analytics/", "frozen code"],
  ["sportvu/", "frozen code"],
  ["PLAN.md", "the plan"],
  ["CLAUDE.md", "the working rules"],
  [".claude/", "the agent settings"],
]);

function under(file, prefix) {
  return prefix.endsWith("/") ? file.startsWith(prefix) : file === prefix;
}

/** The files of a change a candidate may not make, each with the reason. */
export function forbiddenChanges(files) {
  const problems = [];
  for (const file of files) {
    const normal = file.replace(/\\/g, "/").replace(/^\.\//, "");
    const hit = PROTECTED.find(([prefix]) => under(normal, prefix));
    if (hit) problems.push(`${normal}: protected (${hit[1]})`);
    else if (!ALLOWED.some((prefix) => under(normal, prefix))) problems.push(`${normal}: outside the site folders a candidate may change`);
  }
  return problems;
}

function git(args, okStatuses = [0], root = REPO_ROOT) {
  const run = spawnSync("git", args, { cwd: root, encoding: "utf8" });
  if (!okStatuses.includes(run.status)) throw new Error(`git ${args.join(" ")} failed: ${(run.stderr || run.stdout).trim()}`);
  return run.stdout;
}

// Calls a candidate may not remove from the pages, each with what it does.
export const REQUIRED_CALLS = Object.freeze([
  ["checkedText(", "the page's number check of machine-written texts (lib/genai.ts)"],
]);
const CALL_DIRS = ["web/app", "web/components"];

/** How often `needle` occurs in web/app and web/components: at a git ref, or in the working tree when ref is null. */
export function callCount(needle, ref = null, root = REPO_ROOT) {
  // git grep answers 1 when nothing matches.
  const out = git(["grep", "-F", "-c", needle, ...(ref ? [ref] : []), "--", ...CALL_DIRS], [0, 1], root);
  return out.split("\n").filter(Boolean).reduce((sum, line) => sum + Number(line.split(":").pop()), 0);
}

/** The required calls the working tree (of the repository at `root`) holds fewer of than `base`, as problems. */
export function droppedChecks(base, root = REPO_ROOT) {
  const problems = [];
  for (const [needle, what] of REQUIRED_CALLS) {
    const before = callCount(needle, base, root);
    const now = callCount(needle, null, root);
    if (now < before) problems.push(`${needle} calls fell from ${before} to ${now} (${what})`);
  }
  return problems;
}

/**
 * Every file HEAD, the index or the working tree changes since `base`: committed, staged, unstaged and new
 * untracked files. A rename counts as both of its paths.
 */
export function changedFiles(base) {
  const names = (out) => out.split("\n").map((line) => line.trim()).filter(Boolean);
  const files = new Set([
    ...names(git(["diff", "--name-only", "--no-renames", `${base}...HEAD`])),
    ...names(git(["diff", "--name-only", "--no-renames", "HEAD"])),
    ...names(git(["ls-files", "--others", "--exclude-standard"])),
  ]);
  return [...files].sort();
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const index = process.argv.indexOf("--base");
  const base = index > 0 ? process.argv[index + 1] : undefined;
  if (!base) {
    console.error("usage: node scripts/lib/polish-guard.mjs --base <git ref>");
    process.exit(2);
  }
  const files = changedFiles(base);
  const problems = [...forbiddenChanges(files), ...droppedChecks(base)];
  for (const line of problems) console.error(`polish-guard: ${line}`);
  if (problems.length) {
    console.error(`polish-guard: the change since ${base} touches ${problems.length} path(s) a candidate may not change`);
    process.exit(1);
  }
  console.log(`polish-guard: ${files.length} changed file(s) since ${base}, all inside the site folders`);
}
