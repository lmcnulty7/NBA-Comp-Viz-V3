// Every number on the methods page comes from methods.json (CLAUDE.md rule 8; PLAN.md D72).
//
// export/research_methods.py traces every number of methods.json to a report or a lane dossier before it writes
// the file. This check closes the chain on the built site: it reads the main content of out/methods/index.html
// (header, navigation and footer are left out; the footer's build id and schema version come from the
// manifest) and fails when
//   - a number on the page is not a number of out/data/methods.json (a number of a text field, or a numeric
//     field written the way the page writes it). The minus sign, a leading plus and thousands commas are read
//     as the data writes them, so "-0.00052" and "+0.00067" and "101,853" match the stored values;
//   - a lane or an experiment of methods.json is missing from the page;
//   - the page or the data file is missing.
// A number is read with the pattern of export.schemas.NUMBER_TEXT_RE: a digit inside a name (p50, E1, r23,
// gsw_phx_2016) is not a number.
//
//   node scripts/check-methods.mjs [--dir <out dir>]
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { OUT_DIR, shown } from "./lib/paths.mjs";

export const NUMBER_RE = /(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?/g;
const MINUS = String.fromCodePoint(0x2212);
const ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " " };

/** A number as the data writes it: the true minus as a hyphen, no leading plus, no thousands commas. */
export function normalise(token) {
  return token.replace(MINUS, "-").replace(/^\+/, "").replace(/,/g, "");
}

/** The numbers of a text, normalised. */
export function numbersIn(text) {
  return [...text.replaceAll(MINUS, "-").matchAll(NUMBER_RE)].map((match) => normalise(match[0]));
}

/** A stored number written out in plain decimals, the way the page shows a value as stored. */
function plain(value) {
  return value.toLocaleString("en-US", { useGrouping: false, maximumFractionDigits: 20 });
}

/** Every number of a data document: those inside its strings and its numeric fields. */
export function dataNumbers(document) {
  const found = new Set();
  const walk = (value) => {
    if (typeof value === "string") for (const token of numbersIn(value)) found.add(token);
    else if (typeof value === "number") found.add(normalise(plain(value)));
    else if (Array.isArray(value)) value.forEach(walk);
    else if (value && typeof value === "object") Object.values(value).forEach(walk);
  };
  walk(document);
  return found;
}

function decode(text) {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (whole, name) => {
    if (name[0] === "#") {
      const code = name[1] === "x" || name[1] === "X" ? parseInt(name.slice(2), 16) : Number(name.slice(1));
      return String.fromCodePoint(code);
    }
    return ENTITIES[name.toLowerCase()] ?? whole;
  });
}

/** The visible text of the main element of a built page: scripts, styles and tags removed, entities decoded. */
export function mainText(html) {
  const start = html.search(/<main[\s>]/);
  const end = html.lastIndexOf("</main>");
  if (start < 0 || end < start) return null;
  const body = html
    .slice(start, end)
    .replace(/<(script|style|template|noscript)[\s\S]*?<\/\1>/gi, " ")
    .replace(/<[^>]+>/g, " ");
  return decode(body).replace(/\s+/g, " ");
}

/** The problems of a built methods page against its data file. */
export function methodsProblems(html, document) {
  const text = mainText(html);
  if (text === null) return ["methods/index.html has no main element"];
  const known = dataNumbers(document);
  const problems = [];
  const loose = new Set(numbersIn(text).filter((token) => !known.has(token)));
  for (const token of loose) problems.push(`the number ${token} on the page is not in methods.json`);
  for (const lane of document.lanes ?? []) {
    if (!html.includes(`id="lane-${lane.lane}"`)) problems.push(`lane ${lane.lane} of methods.json is not on the page`);
    for (const experiment of lane.experiments ?? []) {
      if (!html.includes(`id="${experiment.id}"`)) {
        problems.push(`experiment ${experiment.id} of methods.json is not on the page`);
      }
    }
  }
  return problems;
}

export function checkMethods(outDir = OUT_DIR) {
  const page = path.join(outDir, "methods", "index.html");
  const data = path.join(outDir, "data", "methods.json");
  for (const file of [page, data]) {
    if (!existsSync(file)) return { problems: [`${shown(file)} does not exist: build the site first`], numbers: 0 };
  }
  const html = readFileSync(page, "utf8");
  const document = JSON.parse(readFileSync(data, "utf8"));
  return { problems: methodsProblems(html, document), numbers: numbersIn(mainText(html) ?? "").length };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const flag = process.argv.indexOf("--dir");
  const dir = flag > 0 ? path.resolve(process.argv[flag + 1]) : OUT_DIR;
  const { problems, numbers } = checkMethods(dir);
  for (const line of problems) console.error(`methods: ${line}`);
  if (problems.length) {
    console.error(`methods: ${problems.length} problem(s) in ${shown(dir)}`);
    process.exitCode = 1;
  } else {
    console.log(`methods: all ${numbers} numbers on the methods page are numbers of methods.json`);
  }
}
