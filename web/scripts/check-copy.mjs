// Site copy holds words, never numbers (PLAN.md: no numeric literals in copy; CLAUDE.md rule 8).
//
// Copy lives in web/content: JSON files of strings, and Markdown files for longer text. This check fails on
//   - any digit in a string or in a Markdown file: a number reaches a page only from the data root, through a
//     {placeholder} that a component fills (lib/copy.ts, lib/format.ts);
//   - a JSON value that is not a string (a number, true, false, null): copy files hold text only;
//   - a placeholder that is not {letters};
//   - an em dash or an en dash;
//   - a phrase that calls the ghost defenders the best or the optimal defense (they are an average defense).
// The other half of the rule is in eslint.config.mjs: a page or a component may not hold literal text, so
// copy cannot be typed anywhere else.
//
//   node scripts/check-copy.mjs [--dir <content dir>]
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { CONTENT_DIR, shown } from "./lib/paths.mjs";

const DIGIT = /\p{Nd}/u;
// The em dash and the en dash, by code point, so this file holds neither.
const DASH = new RegExp(`[${String.fromCodePoint(0x2013, 0x2014)}]`);
const PLACEHOLDER = /\{[^{}]*\}/g;
const GOOD_PLACEHOLDER = /^\{[a-zA-Z]+\}$/;
const BANNED = [
  [/\b(best|optimal|ideal|perfect|correct)\s+(defen[cs]e|defenders?|positions?|positioning)\b/i,
    "ghost defenders are an average defense, never the best or the optimal one"],
];

function excerpt(text) {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > 90 ? `${flat.slice(0, 87)}...` : flat;
}

export function checkText(text, where) {
  const problems = [];
  if (DIGIT.test(text)) problems.push(`${where}: a digit in copy: "${excerpt(text)}"`);
  if (DASH.test(text)) problems.push(`${where}: an em dash or an en dash: "${excerpt(text)}"`);
  for (const placeholder of text.match(PLACEHOLDER) ?? []) {
    if (!GOOD_PLACEHOLDER.test(placeholder)) problems.push(`${where}: placeholder ${placeholder} must be {letters}`);
  }
  for (const [pattern, why] of BANNED) {
    if (pattern.test(text)) problems.push(`${where}: ${why}: "${excerpt(text)}"`);
  }
  return problems;
}

function checkJson(value, where) {
  if (typeof value === "string") return checkText(value, where);
  if (Array.isArray(value)) return value.flatMap((item, index) => checkJson(item, `${where}[${index}]`));
  if (value !== null && typeof value === "object") {
    return Object.entries(value).flatMap(([key, child]) => [
      ...checkText(key, `${where}.${key} (key)`).filter((line) => !line.includes("placeholder")),
      ...checkJson(child, `${where}.${key}`),
    ]);
  }
  return [`${where}: ${JSON.stringify(value)} is not text: a copy file holds strings only`];
}

function contentFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith(".")) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...contentFiles(full));
    else out.push(full);
  }
  return out.sort();
}

export function checkCopy(dir = CONTENT_DIR) {
  if (!existsSync(dir) || !statSync(dir).isDirectory()) return { problems: [`${shown(dir)}: no content folder`], files: 0 };
  const problems = [];
  const files = contentFiles(dir);
  for (const file of files) {
    const name = path.relative(dir, file);
    const text = readFileSync(file, "utf8");
    if (file.endsWith(".json")) {
      try {
        problems.push(...checkJson(JSON.parse(text), name));
      } catch (error) {
        problems.push(`${name}: not valid JSON: ${error.message}`);
      }
    } else if (file.endsWith(".md")) {
      text.split("\n").forEach((line, index) => problems.push(...checkText(line, `${name}:${index + 1}`)));
    } else {
      problems.push(`${name}: copy files are .json or .md`);
    }
  }
  return { problems, files: files.length };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const flag = process.argv.indexOf("--dir");
  const dir = flag > 0 ? path.resolve(process.argv[flag + 1]) : CONTENT_DIR;
  const { problems, files } = checkCopy(dir);
  for (const line of problems) console.error(`copy: ${line}`);
  if (problems.length) {
    console.error(`copy: ${problems.length} problem(s) in ${shown(dir)}`);
    process.exitCode = 1;
  } else {
    console.log(`copy: ${files} file(s) in ${shown(dir)} hold no digit, no dash and no banned phrase`);
  }
}
