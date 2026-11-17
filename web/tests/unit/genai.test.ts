import { expect, test } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { StaticSource } from "@/lib/data/source";
import {
  formatClock,
  formatCount,
  formatNumber,
  formatPercent,
  formatProbability,
  formatSeconds,
  formatSigned,
} from "@/lib/format";
import {
  checkDirections,
  checkLanes,
  checkNumbers,
  checkedText,
  numbersIn,
  playerFacts,
  playerNames,
  possessionFacts,
  possessionNames,
  type Fact,
} from "@/lib/genai";
import { showValue } from "@/lib/tokens";
import type { GeneratedTexts } from "@/lib/types/contract";

/**
 * The page's side of the number check of machine-written texts (PLAN.md P3). The cases of
 * tests/fixtures/genai_cases.json are the ones export/genai.py passes too (tests/test_export_genai.py), so the two
 * checkers agree. On the data root (SITE_DATA_DIR, else web/data) every stored text must pass this check with the
 * facts the page builds from its own data, which also holds the Python mirror of lib/format.ts to the real one.
 */
const WEB_DIR = path.resolve(__dirname, "..", "..");
const CASES = JSON.parse(readFileSync(path.join(WEB_DIR, "..", "tests", "fixtures", "genai_cases.json"), "utf8")) as {
  numbers: { name: string; text: string; facts: Fact[]; names: string[]; ok: boolean; problem: string }[];
  format: { fn: string; value: number; digits?: number; shown: string }[];
  lanes: { name: string; text: string; names: string[]; ok: boolean; lane: string }[];
  directions: { name: string; text: string; facts: Fact[]; names: string[]; ok: boolean; problem: string }[];
};
const named = (process.env.SITE_DATA_DIR ?? "").trim();
const ROOT = named ? path.resolve(WEB_DIR, named) : path.join(WEB_DIR, "data");

for (const item of CASES.numbers) {
  test(`number check: ${item.name}`, () => {
    const problems = checkNumbers(item.text, item.facts, item.names);
    if (item.ok) expect(problems).toEqual([]);
    else expect(problems.some((line) => line.includes(item.problem)), JSON.stringify(problems)).toBe(true);
  });
}

for (const item of CASES.lanes) {
  test(`lane words: ${item.name}`, () => {
    const problems = checkLanes(item.text, item.names);
    if (item.ok) expect(problems).toEqual([]);
    else expect(problems.some((line) => line.includes(`the ${item.lane} lane`)), JSON.stringify(problems)).toBe(true);
  });
}

for (const item of CASES.directions) {
  test(`words of change: ${item.name}`, () => {
    const problems = checkDirections(item.text, item.facts, item.names);
    if (item.ok) expect(problems).toEqual([]);
    else expect(problems.some((line) => line.includes(item.problem)), JSON.stringify(problems)).toBe(true);
  });
}

test("a text whose word of change disagrees with its values is not shown", () => {
  const facts: Fact[] = [
    { field: "epv_start", shown: "0.96", unit: "points", kind: "point", group: "epv_start" },
    { field: "epv.value.0", shown: "0.963", unit: "points", kind: "point", group: "epv.value.0" },
  ];
  const texts = {
    items: [{ key: "a/1", text: "Expected points were 0.96, a point estimate. They dipped to 0.963, an estimate.", numbers: [] }],
  } as unknown as GeneratedTexts;
  expect(checkedText(texts, "a/1", facts, [])).toBeNull();
});

test("a text that names a lane that did not work is not shown", () => {
  const facts: Fact[] = [{ field: "xfg", shown: "0.459", unit: "none", kind: "point", group: "xfg" }];
  const texts = {
    items: [{ key: "a/1", text: "He won his matchup. The shot had a 0.459 chance, a point estimate.", numbers: [] }],
  } as unknown as GeneratedTexts;
  expect(checkedText(texts, "a/1", facts, [])).toBeNull();
});

test("the page formatters give the shown values export/genai.py expects", () => {
  const formatters: Record<string, (value: number, digits?: number) => string> = {
    number: (value, digits) => formatNumber(value, digits),
    signed: (value, digits) => formatSigned(value, digits),
    percent: (value, digits) => formatPercent(value, digits),
    count: (value) => formatCount(value),
    probability: (value) => formatProbability(value),
    clock: (value) => formatClock(value),
    seconds: (value) => formatSeconds(value),
    stored: (value) => showValue(value, "points, as stored"),
    integer: (value) => String(value),
  };
  for (const item of CASES.format) {
    expect(formatters[item.fn]?.(item.value, item.digits), `${item.fn}(${item.value})`).toBe(item.shown);
  }
});

test("every digit of a text belongs to a number", () => {
  const text = "Values 1 and 22.5 and 3,000 and 7:38 and 4th and x9 and -2 and 50% and 0.5 percent.";
  const tokens = numbersIn(text);
  expect(tokens.map((token) => token.canon)).toEqual(["1", "22.5", "3000", "7:38", "4", "9", "-2", "50%", "0.5%"]);
});

test("a text whose number is changed is not shown", () => {
  const facts: Fact[] = [{ field: "xfg", shown: "0.459", unit: "none", kind: "point", group: "xfg" }];
  const texts = {
    items: [{ key: "a/1", text: "The shot had a 0.459 chance, a point estimate.", numbers: [] }],
  } as unknown as GeneratedTexts;
  expect(checkedText(texts, "a/1", facts, [])?.key).toBe("a/1");
  expect(checkedText(texts, "a/2", facts, [])).toBeNull();
  texts.items[0]!.text = "The shot had a 0.46 chance, a point estimate.";
  expect(checkedText(texts, "a/1", facts, [])).toBeNull();
  expect(checkedText(null, "a/1", facts, [])).toBeNull();
});

async function readRoot(relative: string): Promise<unknown> {
  const file = path.join(ROOT, relative);
  if (!existsSync(file)) return undefined;
  return JSON.parse(await readFile(file, "utf8")) as unknown;
}

test("every machine-written text of the data root passes the page's own number check", async () => {
  test.skip(!existsSync(path.join(ROOT, "manifest.json")), "no data root");
  const source = new StaticSource(readRoot);
  const problems: string[] = [];
  let checked = 0;
  const scouting = await source.texts("scouting");
  if (scouting) {
    const players = await source.players();
    for (const item of scouting.items) {
      const player = players.players.find((candidate) => candidate.slug === item.key);
      if (!player) {
        problems.push(`${item.key}: no such player`);
        continue;
      }
      checked += 1;
      const facts = playerFacts(player);
      const names = playerNames(player);
      for (const line of [...checkNumbers(item.text, facts, names), ...checkDirections(item.text, facts, names), ...checkLanes(item.text, names)]) {
        problems.push(`${item.key}: ${line}`);
      }
    }
  }
  const possessions = await source.texts("possession");
  if (possessions) {
    for (const item of possessions.items) {
      const [slug, number] = item.key.split("/") as [string, string];
      const index = await source.possessionIndex(slug);
      const summary = index.possessions.find((row) => String(row.possession) === number);
      if (!summary) {
        problems.push(`${item.key}: no such possession`);
        continue;
      }
      const detail = await source.possession(slug, summary.possession);
      checked += 1;
      const facts = possessionFacts(summary, detail);
      const names = possessionNames(detail);
      for (const line of [
        ...checkNumbers(item.text, facts, names),
        ...checkDirections(item.text, facts, names),
        ...checkLanes(item.text, names),
      ]) {
        problems.push(`${item.key}: ${line}`);
      }
    }
  }
  expect(problems).toEqual([]);
  test.info().annotations.push({ type: "texts", description: String(checked) });
});

test("the reading rules refuse what site copy refuses in a generated text", async () => {
  const { generatedTextProblems } = (await import("../../scripts/lib/generated-text.mjs")) as unknown as {
    generatedTextProblems(text: string): string[];
  };
  const plain =
    "The offense had the ball for 17.0 seconds. The expected points started at 0.89. The last shot had a 0.459 " +
    "chance to go in. It missed, so the offense scored 0 points. Each value is a point estimate.";
  expect(generatedTextProblems(plain)).toEqual([]);
  const dash = String.fromCodePoint(0x2014);
  expect(generatedTextProblems(`${plain} It was close ${dash} very close.`).join()).toContain("dash");
  expect(generatedTextProblems(`${plain} The xFG was low.`).join()).toContain("jargon");
  expect(generatedTextProblems(`${plain} That is the best defense.`).join()).toContain("forbidden");
  expect(generatedTextProblems("Too short.").join()).toContain("words");
  const hard =
    "Comprehensive institutional probabilistic considerations notwithstanding, organizational infrastructure " +
    "deteriorated considerably throughout intercontinental collaborations yesterday.";
  expect(generatedTextProblems(`${plain} ${hard}`).join()).toContain("grade");
});

test("the private build refuses a data root whose text the contract's own rules refuse", async () => {
  const { sampleRoot } = await import("../helpers/scripts");
  const { contractRuleProblems } = (await import("../../scripts/build.mjs")) as unknown as {
    contractRuleProblems(root: string, allowMissing?: string[]): string[];
  };
  const root = await sampleRoot();
  expect(contractRuleProblems(root)).toEqual([]);
  // A lane word passes the page's number check and the JSON Schema; only the contract's own rules refuse it.
  const file = path.join(root, "genai", "possession.json");
  const doc = JSON.parse(readFileSync(file, "utf8")) as GeneratedTexts;
  doc.items[0]!.text = `${doc.items[0]!.text} The team won the matchup.`;
  const { writeFileSync } = await import("node:fs");
  const { createHash } = await import("node:crypto");
  const text = `${JSON.stringify(doc)}\n`;
  writeFileSync(file, text);
  const manifestFile = path.join(root, "manifest.json");
  const manifest = JSON.parse(readFileSync(manifestFile, "utf8")) as { files: { path: string; bytes: number; sha256: string }[] };
  const entry = manifest.files.find((candidate) => candidate.path === "genai/possession.json")!;
  entry.bytes = Buffer.byteLength(text);
  entry.sha256 = createHash("sha256").update(text).digest("hex");
  writeFileSync(manifestFile, `${JSON.stringify(manifest)}\n`);
  const problems = contractRuleProblems(root);
  expect(problems.join("\n")).toContain("matchups lane");
});
