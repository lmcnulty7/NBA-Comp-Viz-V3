// WCAG AA contrast of the design tokens, in the light and the dark theme.
//
//   text: 4.5 to 1 against its background (WCAG 2 success criterion 1.4.3)
//   mark: 3 to 1 against its background (1.4.11): court marks, the focus ring, the marker
//
// Every pair the site uses is listed here. A new token that carries text or a mark gets a line.
//   node scripts/check-contrast.mjs [--tokens <file>]
import { readFileSync } from "node:fs";
import { TOKENS_FILE, shown } from "./lib/paths.mjs";
import { contrast, resolve, themes } from "./lib/tokens.mjs";

const TEXT = 4.5;
const MARK = 3;

// [foreground token, background token, minimum ratio, what it is]
export const PAIRS = [
  ["--ink", "--ground", TEXT, "body text"],
  ["--muted", "--ground", TEXT, "secondary text"],
  ["--faint", "--ground", TEXT, "captions and labels"],
  ["--ink", "--wash", TEXT, "text on the wash"],
  ["--muted", "--wash", TEXT, "secondary text on the wash"],
  ["--invert-ink", "--invert-ground", TEXT, "the fixture banner"],
  ["--accent", "--ground", TEXT, "accent text and link hover"],
  ["--offense-text", "--ground", TEXT, "offense label"],
  ["--defense-text", "--ground", TEXT, "defense label"],
  ["--ghost-text", "--ground", TEXT, "ghost label"],
  ["--truth-text", "--ground", TEXT, "truth label"],
  ["--pass", "--ground", TEXT, "status text: met, works"],
  ["--warn", "--ground", TEXT, "status text: inconclusive"],
  ["--fail", "--ground", TEXT, "status text: not met, failed"],
  ["--offense-on", "--offense", TEXT, "jersey number on an offense disc"],
  ["--defense-on", "--defense", TEXT, "jersey number on a defense disc"],
  ["--ground", "--muted", TEXT, "jersey number on a neutral disc"],
  ["--offense", "--ground", MARK, "offense disc"],
  ["--defense", "--ground", MARK, "defense disc"],
  ["--muted", "--ground", MARK, "neutral disc"],
  ["--ghost", "--ground", MARK, "ghost ring"],
  ["--truth", "--ground", MARK, "truth mark"],
  ["--ball", "--ground", MARK, "ball"],
  ["--accent", "--ground", MARK, "marker, focus ring, link underline"],
  ["--court-line", "--ground", MARK, "court lines"],
  ["--ink", "--ground", MARK, "rules that carry meaning (table head, figure top)"],
];

export function checkContrast(css) {
  const { light, dark, darkBlocksDiffer } = themes(css);
  const problems = darkBlocksDiffer.map(
    (name) => `the two dark blocks of tokens.css give ${name} different values`,
  );
  const rows = [];
  for (const [themeName, tokens] of [["light", light], ["dark", dark]]) {
    for (const [fg, bg, minimum, what] of PAIRS) {
      let ratio;
      try {
        ratio = contrast(resolve(tokens, fg), resolve(tokens, bg));
      } catch (error) {
        problems.push(`${themeName}: ${fg} on ${bg}: ${error.message}`);
        continue;
      }
      // The exact ratio is compared, never the rounded one: 4.499 to 1 does not meet 4.5 to 1.
      const ok = ratio >= minimum;
      rows.push({ theme: themeName, fg, bg, minimum, ratio, ok, what });
      if (!ok) {
        problems.push(`${themeName}: ${fg} on ${bg} is ${ratio.toFixed(2)} to 1, below ${minimum} to 1 (${what})`);
      }
    }
  }
  return { rows, problems };
}

function main() {
  const flag = process.argv.indexOf("--tokens");
  const file = flag > 0 ? process.argv[flag + 1] : TOKENS_FILE;
  const { rows, problems } = checkContrast(readFileSync(file, "utf8"));
  for (const row of rows) {
    const mark = row.ok ? "ok  " : "FAIL";
    console.log(
      `${mark} ${row.theme.padEnd(5)} ${row.ratio.toFixed(2).padStart(5)}  (min ${row.minimum})  ` +
        `${row.fg} on ${row.bg}: ${row.what}`,
    );
  }
  if (problems.length) {
    for (const line of problems) console.error(`contrast: ${line}`);
    console.error(`contrast: ${problems.length} problem(s) in ${shown(file)}`);
    process.exitCode = 1;
    return;
  }
  console.log(`contrast: ${rows.length} pairs pass WCAG AA in both themes (${shown(file)})`);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
