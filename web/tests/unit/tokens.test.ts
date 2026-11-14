import { expect, test } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { previewLanes, scorecardBars, targetShareOf } from "@/components/coldopen/model";
import { storyData, storyLines, storyScenes } from "@/components/coldopen/story";
import { copy } from "@/lib/copy";
import { MINUS } from "@/lib/format";
import {
  OPTIONAL_TOKENS,
  TOKENS,
  TokenError,
  canRender,
  renderLine,
  renderTokens,
  resolveToken,
  shiftPoint,
  showValue,
  storedDigits,
  tokensIn,
  walkField,
  type TokenData,
} from "@/lib/tokens";
import type { DemoFrames, DemoSwap, LaneResult, Manifest, PlayersFile, Results, Scorecard } from "@/lib/types/contract";
import { WEB_DIR } from "../helpers/scripts";

interface ScriptModule {
  scriptTokens(): Map<string, { file: string; field: string; shownAs: string; copiedFrom: string }>;
}

/** The data root the checks run on: SITE_DATA_DIR, else web/data (scripts/lib/paths.mjs). */
function dataRoot(): string {
  const named = (process.env.SITE_DATA_DIR ?? "").trim();
  return named ? path.resolve(WEB_DIR, named) : path.join(WEB_DIR, "data");
}

function readData<T>(relative: string): T | null {
  const file = path.join(dataRoot(), relative);
  return existsSync(file) ? (JSON.parse(readFileSync(file, "utf8")) as T) : null;
}

function siteData(): TokenData {
  return {
    manifest: readData<Manifest>("manifest.json")!,
    scorecard: readData<Scorecard>("scorecard.json")!,
    players: readData<PlayersFile>("players.json")!,
    swap: readData<DemoSwap>("demo/swap.json"),
    frames: readData<DemoFrames>("demo/frames.json"),
  };
}

test("the token table of the site is the token table of docs/SCRIPT.md, row for row", async () => {
  const { scriptTokens } = (await import("../../scripts/lib/script-doc.mjs")) as unknown as ScriptModule;
  const table = scriptTokens();
  expect(table.size).toBeGreaterThan(20);
  const fromScript = Object.fromEntries(
    [...table].map(([token, row]) => [token, { file: row.file, field: row.field, shownAs: row.shownAs }]),
  );
  expect(TOKENS).toEqual(fromScript);
});

test("a stored number keeps its digits: the point moves, nothing is rounded or multiplied", () => {
  expect(storedDigits(0.251)).toBe("0.251");
  expect(storedDigits(1.84)).toBe("1.84");
  expect(storedDigits(1198)).toBe("1198");
  expect(storedDigits(1e-7)).toBe("0.0000001");
  expect(storedDigits(-0.67)).toBe("-0.67");
  expect(shiftPoint("0.251", 2)).toBe("25.1");
  expect(shiftPoint("0.819", 2)).toBe("81.9");
  expect(shiftPoint("0.25", 2)).toBe("25");
  expect(shiftPoint("0.2", 2)).toBe("20");
  expect(shiftPoint("1", 2)).toBe("100");
  expect(shiftPoint("0.001", 2)).toBe("0.1");
  expect(shiftPoint("0.031245", 2)).toBe("3.1245");
  expect(shiftPoint("-0.05", 2)).toBe("-5");
  // 0.07 * 100 is 7.000000000000001 in floating point; the page never shows that.
  expect(showValue(0.07, "share as a percentage")).toBe("7");
  expect(showValue(0.035, "probability as percentage points")).toBe("3.5");
  expect(showValue(-0.67, "feet, as stored")).toBe(`${MINUS}0.67`);
  expect(showValue(9500, "count, as stored")).toBe("9,500");
  expect(showValue(1.92, "feet, as stored")).toBe("1.92");
  expect(showValue(21.39, "as stored")).toBe("21.39");
});

test("a field path selects exactly one value, and a wrong name fails", () => {
  const card = {
    builds: [
      { role: "adopted", rows: [{ key: "position_error", cells: [{ split: "heldout_game", values: { p50_ft: 1.9 } }] }] },
      { role: "candidate", rows: [] },
    ],
  };
  const good = "builds[role=adopted].rows[key=position_error].cells[split=heldout_game].values[p50_ft]";
  expect(walkField(card, good)).toBe(1.9);
  for (const bad of [
    good.replace("p50_ft", "p50"),
    good.replace("position_error", "position"),
    good.replace("role=adopted", "role=nobody"),
    "builds.rows",
  ]) {
    expect(() => walkField(card, bad), bad).toThrow(TokenError);
  }
  expect(walkField({ scenes: [{ frames: [{ error_ft: 1 }, { error_ft: 2 }] }] }, "scenes[0].frames[k].error_ft", { frame: 1 })).toBe(2);
  expect(() => walkField({ scenes: [{ frames: [] }] }, "scenes[0].frames[k].error_ft")).toThrow(TokenError);
  // A missing last key is an error unless the caller marks it optional; a null value reads as undefined.
  expect(() => walkField({ n_windows: 3 }, "fill_share")).toThrow(TokenError);
  expect(() => walkField({ summaries: [{ kind: "xfg" }] }, "summaries[kind=xfg].mean_abs_diff")).toThrow(TokenError);
  expect(walkField({ scenes: [{}] }, "scenes[0].p50_error_ft", {}, { optional: true })).toBeUndefined();
  expect(walkField({ scenes: [{ p50_error_ft: null }] }, "scenes[0].p50_error_ft")).toBeUndefined();
  // Optional covers the last key only, never a step on the way.
  expect(() => walkField({}, "scenes[0].p50_error_ft", {}, { optional: true })).toThrow(TokenError);
});

test("a token whose field is missing from its file throws, unless docs/SCRIPT.md gives it a fallback", () => {
  const data = siteData();
  test.skip(!data.swap || !data.frames, "this data root has no demo files");
  const swap = structuredClone(data.swap!) as unknown as Record<string, unknown>;
  delete swap.fill_share;
  expect(() => resolveToken("swap.fill_share", { ...data, swap: swap as unknown as DemoSwap })).toThrow(TokenError);
  const frames = structuredClone(data.frames!);
  delete frames.scenes[0]!.p50_error_ft;
  for (const frame of frames.scenes[0]!.frames) delete frame.error_ft;
  expect(resolveToken("frames.scene.p50_error_ft", { ...data, frames })).toBeNull();
  expect(resolveToken("frames.scene.frame.error_ft", { ...data, frames }, { frame: 0 })).toBeNull();
  expect([...OPTIONAL_TOKENS].every((token) => token in TOKENS)).toBe(true);
});

test("an unknown token fails, so a page that uses one fails the build", () => {
  const data = siteData();
  expect(() => resolveToken("swap.fill_shares", data)).toThrow(/not a token of docs\/SCRIPT\.md/);
  expect(() => renderTokens("It is {swap.mean_abs} points.", data)).toThrow(TokenError);
  expect(() => renderLine("It is {scorecard.adopted.heldout_game.typo} feet.", "fallback", data)).toThrow(TokenError);
  expect(tokensIn("A {letters} placeholder is not a token, {swap.fill_share} is.")).toEqual(["swap.fill_share"]);
});

test("every token of the script resolves on the data root, and has a value wherever its data is present", () => {
  const data = siteData();
  const frames = data.frames?.scenes[0]?.frames ?? [];
  for (const [token, spec] of Object.entries(TOKENS)) {
    const value = resolveToken(token, data, { frame: 0 });
    if (value !== null) expect(value, token).toMatch(/^[−]?[\d,]+(\.\d+)?$/);
    // Null is allowed only where the copy has a fallback for it: a demo file this build lacks, a model of the
    // comparison whose rebuild did not pass (C6), or a field docs/SCRIPT.md lets be missing (OPTIONAL_TOKENS).
    const filePresent = spec.file === "demo/swap.json" ? data.swap !== null : spec.file === "demo/frames.json" ? data.frames !== null : true;
    const kind = /^summaries\[kind=([a-z]+)\]\./.exec(spec.field)?.[1];
    const summaryAvailable = !kind || data.swap?.summaries.find((summary) => summary.kind === kind)?.available === true;
    if (filePresent && summaryAvailable && !OPTIONAL_TOKENS.has(token)) expect(value, token).not.toBeNull();
  }
  if (frames.length) {
    // The readout token reads the frame on screen.
    frames.forEach((frame, k) => {
      const shown = resolveToken("frames.scene.frame.error_ft", data, { frame: k });
      expect(shown).toBe(frame.error_ft == null ? null : showValue(frame.error_ft, "feet, as stored"));
    });
  }
});

test("a model of the comparison whose check failed shows its fallback line and no number (C6)", () => {
  const data = siteData();
  test.skip(!data.swap, "this data root has no demo comparison");
  const swap = structuredClone(data.swap!);
  const xfg = swap.summaries.find((summary) => summary.kind === "xfg")!;
  xfg.available = false;
  xfg.n = 0;
  // Even a number left behind in a failed summary is not shown.
  const broken: TokenData = { ...data, swap };
  const step = copy.home.coldOpen.swap.steps[2]!;
  expect(canRender(step, broken)).toBe(false);
  expect(renderLine(step, copy.home.coldOpen.swap.fallback, broken)).toBe(copy.home.coldOpen.swap.fallback);
  const lines = storyLines("swap", broken);
  expect(lines).toContain(copy.home.coldOpen.swap.fallback);
  expect(lines.some((line) => line.startsWith("Shot quality is the chance"))).toBe(false);
  expect(lines.join(" ")).not.toContain(showValue(xfg.mean_abs_diff ?? 0, "probability as percentage points") + " percentage");
});

test("the comparison scene states n, the mean absolute difference and the fill share of each model (C6)", () => {
  const data = siteData();
  test.skip(!data.swap, "this data root has no demo comparison");
  const swap = data.swap!;
  const text = storyLines("swap", data).join(" ");
  expect(text).toContain("illustration, not a test");
  expect(text).toContain(`${showValue(swap.fill_share, "share as a percentage")} percent`);
  expect(text).toContain(`on ${showValue(swap.n_windows, "count, as stored")} stretches`);
  const rules = { xfg: "probability as percentage points", epv: "points, as stored", ghosting: "feet, as stored" } as const;
  for (const summary of swap.summaries) {
    if (summary.available && summary.mean_abs_diff != null) {
      expect(text).toContain(`by ${showValue(summary.mean_abs_diff, rules[summary.kind])} `);
      expect(text).toContain(`across ${showValue(summary.n, "count, as stored")} `);
    }
  }
  const unavailable = swap.summaries.filter((summary) => !summary.available).length;
  if (unavailable) expect(text).toContain(copy.home.coldOpen.swap.fallback);
});

test("the gap of the scene is never shown without the scorecard value of the game (D50)", () => {
  const data = siteData();
  const truth = copy.home.coldOpen.truth;
  const p50 = "{scorecard.adopted.heldout_game.position_error.p50_ft}";
  // In the copy: every line with the gap of the stretch carries the scorecard value too.
  for (const line of [...truth.steps, truth.fallback]) {
    if (line.includes("{frames.scene.p50_error_ft}")) expect(line).toContain(p50);
  }
  // In the picture: the scorecard row and the stretch value come together.
  const story = storyData(data, "public", () => null);
  if (story.position?.stretch) expect(story.position.values).toContain(resolveToken(p50.slice(1, -1), data)!);
  const lines = storyLines("truth", data);
  expect(lines.some((line) => line.includes(resolveToken(p50.slice(1, -1), data)!))).toBe(true);
});

test("the public build shows the request-access note and no clip; the private build names the clip", () => {
  const data = siteData();
  const publicScenes = storyScenes(data, "public");
  expect(publicScenes.find((scene) => scene.key === "clip")?.caption).toBe(copy.home.coldOpen.clip.caption);
  expect(storyData(data, "public", () => null).media?.clip ?? null).toBeNull();
  const privateScenes = storyScenes(data, "private");
  expect(privateScenes.every((scene) => !scene.caption)).toBe(true);
  const scene = data.frames?.scenes[0];
  if (scene) expect(storyData(data, "private", () => null).media?.clip).toContain(`clips/${scene.clip}`);
});

test("the three tracking bars read their values and statuses from the scorecard", () => {
  const data = siteData();
  const bars = scorecardBars(data.scorecard);
  const adopted = data.scorecard.builds.find((build) => build.role === "adopted")!;
  for (const bar of bars) {
    const cell = adopted.rows.find((row) => row.key === bar.key)!.cells.find((candidate) => candidate.split === "heldout_game")!;
    const field = bar.key === "team_labels" ? "accuracy" : "rate";
    expect(bar.value).toBe(`${showValue(cell.values[field]!, "share as a percentage")}%`);
  }
  expect(targetShareOf("<= 5%")).toBe(0.05);
  expect(targetShareOf(">= 95%")).toBe(0.95);
  expect(targetShareOf("p50 <= 2.0 ft")).toBeNull();
});

test("the results card shows a failed model among its first rows", () => {
  const lane = (name: string, verdict: LaneResult["verdict_class"]) => ({ lane: name, verdict_class: verdict }) as LaneResult;
  const lanes = [lane("a", "works"), lane("b", "works"), lane("c", "no_signal"), lane("d", "failed")];
  expect(previewLanes(lanes).map((row) => row.lane)).toEqual(["a", "b", "d"]);
  expect(previewLanes([lane("a", "failed"), ...lanes]).map((row) => row.lane)).toEqual(["a", "a", "b"]);
  const results = readData<Results>("results.json");
  if (results?.lanes.some((row) => row.verdict_class === "failed")) {
    expect(previewLanes(results.lanes).some((row) => row.verdict_class === "failed")).toBe(true);
  }
});
