import { expect, test } from "@playwright/test";
import { signedText } from "@/components/results/DataValue";
import { copy } from "@/lib/copy";
import { numberText } from "@/lib/format";
import { repoFileUrl } from "@/lib/repo";
import { walkField, type ShownAs } from "@/lib/tokens";
import type { MethodsFile, ProtocolResult } from "@/lib/types/contract";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");

// The methods and literature page (PLAN.md D72), read against the methods.json the built site serves.

interface Shown {
  field: string;
  shown: string;
  signed: boolean;
  text: string;
}

/**
 * The ends of the ranges that are not valid confidence intervals, read from the result files of the repository
 * the protocol names: the pre-registered game-bootstrap ranges and every game_basic range, as stored and rounded to
 * the page's three decimals, less every number the protocol shows validly. Empty for a fixture build.
 */
function invalidRangeEnds(protocol: ProtocolResult): Set<number> {
  const read = (rel: string): Record<string, unknown> | null => {
    const file = path.join(REPO_ROOT, rel);
    return rel.endsWith(".json") && existsSync(file) ? (JSON.parse(readFileSync(file, "utf8")) as Record<string, unknown>) : null;
  };
  const raw: number[] = [];
  const verdict = (read(protocol.source.path)?.verdict ?? {}) as Record<string, { interval_95?: number[]; stability_lo?: number }>;
  for (const entry of Object.values(verdict)) {
    if (entry && typeof entry === "object") {
      raw.push(...(entry.interval_95 ?? []));
      if (typeof entry.stability_lo === "number") raw.push(entry.stability_lo);
    }
  }
  const intervals = [...protocol.hypotheses, ...protocol.adoption].map((item) => item.interval).find((i) => i);
  const posthoc = intervals ? read(intervals.source.path) : null;
  const statistics = ((posthoc?.valid_intervals as { statistics?: Record<string, { game_basic?: { lo: number; hi: number } }> })
    ?.statistics ?? {}) as Record<string, { game_basic?: { lo: number; hi: number } }>;
  for (const stat of Object.values(statistics)) if (stat.game_basic) raw.push(stat.game_basic.lo, stat.game_basic.hi);
  const ends = new Set<number>([...raw, ...raw.map((x) => Math.round(x * 1000) / 1000)]);
  for (const item of [...protocol.hypotheses, ...protocol.adoption]) {
    if (item.value != null) ends.delete(item.value);
    if (item.interval) {
      ends.delete(item.interval.lo);
      ends.delete(item.interval.hi);
    }
  }
  return ends;
}

async function methodsData(request: import("@playwright/test").APIRequestContext): Promise<MethodsFile> {
  const served = await request.get("data/methods.json");
  expect(served.status()).toBe(200);
  return (await served.json()) as MethodsFile;
}

test("every value on the methods page equals methods.json", async ({ page }) => {
  const methods = await methodsData(page.request);
  await page.goto("methods/");
  const shown = await page.locator('main [data-file="methods.json"]').evaluateAll((nodes) =>
    nodes.map((node) => {
      const element = node as HTMLElement;
      return {
        field: element.dataset.field ?? "",
        shown: element.dataset.shown ?? "",
        signed: element.dataset.signed === "true",
        text: element.textContent ?? "",
      };
    }),
  );
  expect(shown.length).toBeGreaterThan(20);
  const wrong: string[] = [];
  for (const item of shown as Shown[]) {
    const value = walkField(methods, item.field);
    const want =
      item.shown === "text"
        ? String(value)
        : item.shown === "number text"
          ? numberText(String(value))
          : signedText(value as number, item.shown as ShownAs, item.signed);
    if (item.text !== want) wrong.push(`${item.field}: shows ${JSON.stringify(item.text)}, the file holds ${JSON.stringify(want)}`);
  }
  expect(wrong).toEqual([]);
});

test("every lane, paper, benchmark row and test of methods.json is on the page, with its state", async ({ page }) => {
  const methods = await methodsData(page.request);
  await page.goto("methods/");
  await expect(page.locator("h1")).toHaveText(copy.methods.title);
  await expect(page.locator("section[data-lane]")).toHaveCount(methods.lanes.length);
  for (const lane of methods.lanes) {
    const section = page.locator(`section[data-lane="${lane.lane}"]`);
    const papers = section.locator(`a[data-field^="lanes[lane=${lane.lane}].foundations["][data-field$=".cite"]`);
    await expect(papers).toHaveCount(lane.foundations.length);
    for (const [index, paper] of lane.foundations.entries()) await expect(papers.nth(index)).toHaveAttribute("href", paper.url);
    const rows = section.locator("li[data-kind]");
    await expect(rows).toHaveCount(lane.benchmarks.length);
    for (const [index, row] of lane.benchmarks.entries()) {
      await expect(rows.nth(index)).toHaveAttribute("data-kind", row.comparability_class);
      await expect(rows.nth(index)).toContainText(copy.methods.kind[row.comparability_class]);
      // A number of ours shows its interval, or says it is a point estimate or a setting.
      for (const [at, value] of row.ours.entries()) {
        const note = rows.nth(index).locator(`[data-field="lanes[lane=${lane.lane}].benchmarks[${index}].ours[${at}].value"] + .note`);
        if (value.lo == null) {
          await expect(note).toHaveText(value.kind === "setting" ? copy.methods.lane.setting : copy.methods.lane.point);
        } else {
          await expect(note).toHaveAttribute("data-interval", "true");
        }
        const url = repoFileUrl(value.source);
        if (url) await expect(rows.nth(index).locator(`a[data-report="${value.source}"]`).first()).toHaveAttribute("href", url);
      }
    }
    for (const experiment of lane.experiments) {
      const item = section.locator(`li[data-experiment="${experiment.id}"]`);
      await expect(item).toHaveAttribute("data-status", experiment.status);
      await expect(item).toContainText(copy.methods.status[experiment.status]);
    }
  }
  const ran = methods.lanes.flatMap((lane) => lane.experiments.filter((e) => e.status !== "pre_registered"));
  // On real data winprob-E2 is left unscored by name, so the not-scored check below cannot pass by finding no such
  // experiment. The synthetic fixture has no winprob lane.
  if (!ran.some((e) => e.id.startsWith("fixture-"))) {
    expect(ran.find((e) => e.id === "winprob-E2")?.test_not_scored).toBe(true);
  }
  await expect(page.locator("[data-result]")).toHaveCount(ran.length);
  if (ran.length === 0) await expect(page.getByTestId("no-results")).toHaveText(copy.methods.results.none);
  for (const experiment of ran) {
    const result = page.locator(`[data-result="${experiment.id}"]`);
    const parts = experiment.parts ?? [];
    const partStages = parts.flatMap((part) => [part.development, part.test].filter((stage) => stage != null));
    await expect(result.locator("[data-stage]")).toHaveCount(
      (experiment.development ? 1 : 0) + (experiment.test ? 1 : 0) + partStages.length,
    );
    // An experiment read part by part shows every part, each with its own state and its own stages.
    await expect(result.locator("[data-part]")).toHaveCount(parts.length);
    for (const [index, part] of parts.entries()) {
      const shown = result.locator("[data-part]").nth(index);
      await expect(shown).toHaveAttribute("data-part", part.label);
      await expect(shown).toHaveAttribute("data-part-status", part.status);
      await expect(shown).toContainText(copy.methods.results.part[part.status]);
      await expect(shown.locator("[data-stage]")).toHaveCount((part.development ? 1 : 0) + (part.test ? 1 : 0));
      for (const stage of [part.development, part.test]) {
        if (!stage) continue;
        const facts = shown.locator(`[data-stage="${stage.stage}"]`);
        await expect(facts).toHaveAttribute("data-met", stage.met ? "true" : "false");
        await expect(facts.getByTestId("reading")).toHaveCount(stage.reading ? 1 : 0);
        if (stage.reading) await expect(facts.getByTestId("reading")).toContainText(stage.reading);
      }
    }
    if (experiment.status === "development_met_test_pending") {
      await expect(result.getByTestId("test-pending")).toHaveText(copy.methods.results.pending);
    }
    // The line saying the test was kept unused shows for an experiment left unscored (winprob-E2, gravity-E1,
    // matchups-E2) and for a part left unscored, and nowhere else.
    const unscored = (experiment.test_not_scored ? 1 : 0) + (experiment.parts ?? []).filter((p) => p.test_not_scored).length;
    await expect(result.getByTestId("test-not-scored")).toHaveCount(unscored);
    if (unscored > 0) await expect(result.getByTestId("test-not-scored").first()).toHaveText(copy.methods.results.notScored);
    const protocol = experiment.protocol;
    await expect(result.locator("[data-protocol]")).toHaveCount(protocol ? 1 : 0);
    if (protocol) {
      // Each statement shows its reading, and its valid range marked post hoc or the words point estimate. The
      // ranges set up in advance are not valid (PLAN.md D86) and are never on the page.
      const statements = result.locator("[data-hypothesis]");
      await expect(statements).toHaveCount(protocol.hypotheses.length);
      for (const [index, h] of protocol.hypotheses.entries()) {
        await expect(statements.nth(index)).toHaveAttribute("data-reading", h.result);
        await expect(statements.nth(index)).toContainText(copy.methods.results.protocol.result[h.result]);
        const note = statements.nth(index).locator(".note");
        if (h.interval) await expect(note).toContainText(copy.methods.results.protocol.postHoc);
        else if (h.value != null) await expect(note).toHaveText(copy.methods.lane.point);
      }
      await expect(result.locator("[data-protocol]")).toHaveAttribute("data-adopted", protocol.adopted ? "true" : "false");
      if (protocol.caveat) await expect(result.getByTestId("protocol-caveat")).toHaveText(protocol.caveat);
      await expect(result.getByTestId("not-scored").locator("li")).toHaveCount(protocol.not_scored.length);
      // No end of a range that is not a valid interval is on the page (PLAN.md D86), compared as numbers, unless
      // the same number is a value shown validly (0.961 is both a value and a valid end).
      const leaks = invalidRangeEnds(protocol);
      const shown = (await result.innerText()).replaceAll(String.fromCodePoint(0x2212), "-");
      const onPage = [...shown.matchAll(/(?<![\w.])[+-]?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
      expect(onPage.filter((n) => leaks.has(n))).toEqual([]);
    }
    const notes = experiment.writeup_notes ?? [];
    await expect(result.getByTestId("writeup-notes").locator("q")).toHaveCount(notes.length);
    for (const [index, note] of notes.entries()) await expect(result.getByTestId("writeup-notes").locator("q").nth(index)).toHaveText(note);
    for (const stage of [experiment.development, experiment.test]) {
      if (!stage) continue;
      const facts = result.locator(`[data-stage="${stage.stage}"]`);
      const guards = [...(stage.guard ? [stage.guard] : []), ...(stage.guards ?? [])];
      await expect(facts.getByTestId("guard")).toHaveCount(guards.length);
      for (const [index, guard] of guards.entries()) {
        const shown = facts.getByTestId("guard").nth(index);
        await expect(shown.locator("[data-bucket-worse]")).toHaveCount(guard.buckets.length);
        await expect(shown).toHaveAttribute("data-guard-met", String(guard.met));
      }
      await expect(facts.getByTestId("also-read")).toHaveCount((stage.also_read ?? []).length);
      // a reading word the file states (ghosting-E1's D113 reading on Set A) is shown as written
      await expect(facts.getByTestId("reading")).toHaveCount(stage.reading ? 1 : 0);
      if (stage.reading) await expect(facts.getByTestId("reading")).toContainText(stage.reading);
      const conditions = stage.conditions ?? [];
      await expect(facts.locator("[data-condition-met]")).toHaveCount(conditions.length);
      for (const [index, condition] of conditions.entries()) {
        await expect(facts.locator("[data-condition-met]").nth(index)).toHaveAttribute("data-condition-met", String(condition.met));
      }
      await expect(facts).toHaveAttribute("data-met", stage.met ? "true" : "false");
      await expect(facts).toContainText(stage.met ? copy.methods.results.met : copy.methods.results.missed);
      const url = repoFileUrl(stage.source.path);
      if (url) await expect(facts.locator(`a[data-report="${stage.source.path}"]`)).toHaveAttribute("href", url);
    }
  }
});

test("the methods page is in the nav, and its contents links land on each lane", async ({ page }) => {
  const methods = await methodsData(page.request);
  await page.goto("methods/");
  const nav = page.getByRole("navigation", { name: copy.site.nav.label });
  await expect(nav.locator('[aria-current="page"]')).toHaveText(copy.site.nav.methods);
  const last = methods.lanes[methods.lanes.length - 1]!;
  await page.locator(`main nav a[href="#lane-${last.lane}"]`).click();
  await expect(page).toHaveURL(new RegExp(`#lane-${last.lane}$`));
  await expect(page.locator(`#lane-${last.lane} h2`)).toBeInViewport();
});
