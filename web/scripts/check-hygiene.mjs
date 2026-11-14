// Nothing private and no long dash in the web sources (CLAUDE.md rules 9 and 12), and no source file that git
// silently leaves out.
//
// Scans every text file under web/ that git tracks or would track (so node_modules, the build output and the
// data are left out) for an em dash or an en dash, and for a local path, a cloud mount name or an email
// address (the PRIVATE_PATTERNS of export/schemas.py).
//
// It also lists what git ignores under web/ and fails on anything that is not build output, data or local
// state. The repository .gitignore has rules without a leading slash (site/, logs/, runs/, gates/) that hide
// a folder of that name at any depth: a components/site folder would never be committed.
//
//   node scripts/check-hygiene.mjs
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { privateScanner } from "./lib/contract.mjs";
import { CONTRACT_RULES_FILE, REPO_ROOT, WEB_DIR } from "./lib/paths.mjs";

const BINARY = new Set(["woff", "woff2", "ttf", "otf", "png", "jpg", "jpeg", "gif", "webp", "avif", "ico", "pdf"]);
// The em dash and the en dash, by code point, so this file holds neither.
const DASH = new RegExp(`[${String.fromCodePoint(0x2013, 0x2014)}]`);

function sourceFiles() {
  const result = spawnSync(
    "git",
    ["ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", path.relative(REPO_ROOT, WEB_DIR)],
    { cwd: REPO_ROOT, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 },
  );
  if (result.status !== 0) throw new Error(`git ls-files failed: ${result.stderr}`);
  return result.stdout
    .split("\0")
    .filter(Boolean)
    .map((relative) => path.join(REPO_ROOT, relative))
    .filter((file) => existsSync(file) && !BINARY.has(path.extname(file).slice(1).toLowerCase()));
}

// What git may ignore under web/: dependencies, build output, generated data, private clips, test output and
// local state. Anything else that is ignored is a source file that would be missing from a commit.
const EXPECTED_IGNORED = [
  /^web\/node_modules\/$/,
  /^web\/\.next\/$/,
  /^web\/out\/$/,
  /^web\/data\/$/,
  /^web\/public\/data\/$/,
  /^web\/media-private\/$/,
  /^web\/test-results\/$/,
  /^web\/playwright-report\/$/,
  /^web\/next-env\.d\.ts$/,
  /\.tsbuildinfo$/,
  /(^|\/)\.DS_Store$/,
  /(^|\/)\.env[^/]*$/,
  /(^|\/)npm-debug\.log[^/]*$/,
];

/** Paths under web/ that git ignores and that are not on the expected list. */
export function unexpectedlyIgnored() {
  const result = spawnSync(
    "git",
    ["ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z", "--", path.relative(REPO_ROOT, WEB_DIR)],
    { cwd: REPO_ROOT, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 },
  );
  if (result.status !== 0) throw new Error(`git ls-files failed: ${result.stderr}`);
  return result.stdout
    .split("\0")
    .filter(Boolean)
    .filter((relative) => !EXPECTED_IGNORED.some((pattern) => pattern.test(relative)));
}

export function checkHygiene() {
  const findPrivate = privateScanner();
  const problems = unexpectedlyIgnored().map(
    (relative) => `${relative} is ignored by git and would not be committed: rename it (see the note in this script)`,
  );
  const files = sourceFiles();
  for (const file of files) {
    const name = path.relative(REPO_ROOT, file);
    const text = readFileSync(file, "utf8");
    text.split("\n").forEach((line, index) => {
      if (DASH.test(line)) problems.push(`${name}:${index + 1}: an em dash or an en dash`);
    });
    // The contract rules file is the list of private patterns itself.
    if (file === CONTRACT_RULES_FILE) continue;
    for (const found of findPrivate(text)) problems.push(`${name}: private string ${JSON.stringify(found)}`);
  }
  return { problems, files: files.length };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const { problems, files } = checkHygiene();
    for (const line of problems) console.error(`hygiene: ${line}`);
    if (problems.length) {
      console.error(`hygiene: ${problems.length} problem(s)`);
      process.exitCode = 1;
    } else {
      console.log(`hygiene: ${files} source file(s) under web/ hold no long dash and no private string, and none is ignored by git`);
    }
  } catch (error) {
    console.error(`hygiene: ${error.message}`);
    process.exitCode = 1;
  }
}
