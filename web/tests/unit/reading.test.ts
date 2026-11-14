import { expect, test } from "@playwright/test";
import { cpSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { WEB_DIR, runScript, tempDir } from "../helpers/scripts";

interface Readability {
  countSyllables(word: string): number;
  wordsOf(text: string): string[];
  sentenceCount(text: string): number;
  fkGrade(text: string, tokenSyllables?: number): number;
  plainWordProblems(line: string, options?: { allowGroundTruth?: boolean }): string[];
}

interface Slides {
  parseSlide(text: string, name?: string): {
    meta: Record<string, string>;
    title: string;
    blocks: ({ kind: "p"; text: string; fallback?: string } | { kind: "list"; items: string[] })[];
    notes: string[];
  };
}

const load = async () => (await import("../../scripts/lib/readability.mjs")) as unknown as Readability;

test("syllables match the vocabulary checked by hand for docs/SCRIPT.md", async () => {
  const { countSyllables } = await load();
  const checked: Record<string, number> = {
    player: 2, players: 2, playing: 2, trying: 2, frame: 1, frames: 1, these: 1, whole: 1, table: 2, tables: 2,
    boxes: 2, places: 2, moved: 1, trained: 1, wanted: 2, missed: 1, checked: 1, possession: 3, value: 2,
    measured: 2, against: 2, average: 3, illustration: 4, position: 3, quality: 3, broadcast: 2, camera: 3,
    sensor: 2, people: 2, single: 2, league: 1, true: 1, video: 3, arena: 3, data: 2, you: 1, beyond: 2, eye: 1,
    percentage: 3, defender: 3, easier: 3, being: 2, typical: 3, basketball: 3, TV: 2, NBA: 3, "1.92": 1,
  };
  const wrong = Object.entries(checked).filter(([word, want]) => countSyllables(word) !== want);
  expect(wrong).toEqual([]);
});

test("the grade formula gives the known values", async () => {
  const { wordsOf, sentenceCount, fkGrade } = await load();
  const easy = "The cat sat on the mat. The dog ran to the man.";
  expect(wordsOf(easy)).toHaveLength(12);
  expect(sentenceCount(easy)).toBe(2);
  expect(fkGrade(easy)).toBeCloseTo(0.39 * 6 + 11.8 * 1 - 15.59, 9);
  expect(fkGrade("Institutional considerations necessitate comprehensive organizational restructuring immediately.")).toBeGreaterThan(20);
  expect(wordsOf("It is {swap.fill_share} percent.")).toEqual(["It", "is", "TOKEN", "percent"]);
  expect(fkGrade("It is {swap.fill_share} percent.", 4)).toBeGreaterThan(fkGrade("It is {swap.fill_share} percent."));
  expect(sentenceCount("A headline with no full stop\n\nA sentence. And another.")).toBe(3);
});

test("the word list catches jargon and claims the site never makes", async () => {
  const { plainWordProblems } = await load();
  expect(plainWordProblems("The median gap is small.")).toHaveLength(1);
  expect(plainWordProblems("Where the ghost defender stands.")).toHaveLength(1);
  expect(plainWordProblems("The held-out games.")).toHaveLength(1);
  expect(plainWordProblems("This proves it works.")).toHaveLength(1);
  expect(plainWordProblems("Where the best defender stands.")).toHaveLength(1);
  expect(plainWordProblems("Measured against ground truth.", { allowGroundTruth: true })).toEqual([]);
  expect(plainWordProblems("It is {scorecard.adopted.heldout_game.position_error.p50_ft} feet.")).toEqual([]);
});

test("the cold open and the slides pass the reading check", () => {
  const done = runScript("check-reading.mjs");
  expect(done.status, done.output).toBe(0);
  expect(done.output).toContain("scene 5 (swap)");
  expect(done.output).toContain("talk/swap.md");
});

interface Home {
  coldOpen: { map: { steps: string[] }; boxes: { steps: string[] }; truth: { labels: Record<string, string> } };
  ui: Record<string, unknown>;
}

function brokenContent(change: (home: Home) => void, slide?: [string, string]): string {
  const dir = tempDir("reading");
  cpSync(path.join(WEB_DIR, "content"), dir, { recursive: true });
  const file = path.join(dir, "home.json");
  const home = JSON.parse(readFileSync(file, "utf8")) as Home;
  change(home);
  writeFileSync(file, JSON.stringify(home));
  if (slide) writeFileSync(path.join(dir, "talk", `${slide[0]}.md`), slide[1]);
  return dir;
}

for (const [name, change, message, slide] of [
  ["a line changed from the approved script", (home: Home) => (home.coldOpen.map.steps[0] = "A box is a box."), "is not a line of scene 3"],
  ["an approved line dropped", (home: Home) => home.coldOpen.boxes.steps.pop(), "lacks the approved line"],
  ["a label changed", (home: Home) => (home.coldOpen.truth.labels.checked = "frames we checked"), "is not a label of scene 4"],
  ["jargon in the words around the picture", (home: Home) => (home.ui.stretch = "the median of this stretch"), "jargon"],
  ["an unknown token", (home: Home) => (home.ui.players = "{players.n_listed} players."), "is not in the token table"],
  [
    "a slide above grade nine",
    () => undefined,
    "reads at grade",
    ["built", "---\nvisual: none\n---\n# How it was built\n\nInstitutional considerations necessitate comprehensive organizational restructuring immediately.\n"],
  ],
  ["a slide that drops a line of its scene", () => undefined, "lacks the line of scene 3", ["map", "---\nscene: map\nvisual: map\n---\n# Where?\n\nA box on a screen is not a spot on the floor. The camera pans and zooms, so the same pixel is a new place in every frame.\n"]],
] as const) {
  test(`the reading check refuses ${name}`, () => {
    const dir = brokenContent(change as (home: Home) => void, slide as [string, string] | undefined);
    const done = runScript("check-reading.mjs", ["--content", dir]);
    expect(done.output).toContain(message);
    expect(done.status).toBe(1);
  });
}

test("a slide file is read the same way by the site and the check", async () => {
  const { parseSlide } = (await import("../../scripts/lib/slides.mjs")) as unknown as Slides;
  const slide = parseSlide(
    "---\nscene: truth\nvisual: truth\n---\n# Title\n\nFirst line {swap.fill_share}.\n\nOtherwise: The fallback.\n\n- one\n- two\n\n## Notes\n\n- a note\n  that wraps\n",
  );
  expect(slide.meta).toEqual({ scene: "truth", visual: "truth" });
  expect(slide.title).toBe("Title");
  expect(slide.blocks).toEqual([
    { kind: "p", text: "First line {swap.fill_share}.", fallback: "The fallback." },
    { kind: "list", items: ["one", "two"] },
  ]);
  expect(slide.notes).toEqual(["a note that wraps"]);
  expect(() => parseSlide("# A\n\n# B\n")).toThrow(/more than one title/);
  expect(() => parseSlide("---\nvisual: film\n---\n# A\n")).toThrow(/visual/);
  expect(() => parseSlide("Otherwise: nothing before it.\n\n# A\n")).toThrow(/fallback/);
  expect(() => parseSlide("Just text.\n")).toThrow(/no title/);
});
