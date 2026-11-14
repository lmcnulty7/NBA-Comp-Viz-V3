import { expect, test } from "@playwright/test";
import { signedText } from "@/components/results/DataValue";
import { copy } from "@/lib/copy";
import { repoFileUrl } from "@/lib/repo";
import { walkField, type ShownAs } from "@/lib/tokens";
import type { Results, Scorecard } from "@/lib/types/contract";

interface Shown {
  file: string;
  field: string;
  shown: string;
  signed: boolean;
  text: string;
}

test("every value on the results page equals its data file (C5, C7)", async ({ page }) => {
  const files: Record<string, unknown> = {
    "scorecard.json": await (await page.request.get("data/scorecard.json")).json(),
    "results.json": await (await page.request.get("data/results.json")).json(),
  };
  await page.goto("results/");
  const shown = await page.locator("main [data-field]").evaluateAll((nodes) =>
    nodes.map((node) => {
      const element = node as HTMLElement;
      return {
        file: element.dataset.file ?? "",
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
    const value = walkField(files[item.file], item.field);
    const want =
      item.shown === "text" ? String(value) : signedText(value as number, item.shown as ShownAs, item.signed);
    if (item.text !== want) wrong.push(`${item.file} ${item.field}: shows ${JSON.stringify(item.text)}, the file holds ${JSON.stringify(want)}`);
  }
  expect(wrong).toEqual([]);
});

test("the scorecard shows each build's rows with the status the data states, the candidate after the adopted build", async ({ page }) => {
  const scorecard = (await (await page.request.get("data/scorecard.json")).json()) as Scorecard;
  await page.goto("results/");
  const builds = page.locator("[data-build]");
  await expect(builds).toHaveCount(scorecard.builds.length);
  await expect(builds.first()).toHaveAttribute("data-role", "adopted");
  for (const build of scorecard.builds) {
    const section = page.locator(`[data-build="${build.build}"]`);
    await expect(section.locator("tr[data-row]")).toHaveCount(build.rows.length);
    const statuses = await section.locator("td[data-status]").evaluateAll((cells) => cells.map((cell) => cell.getAttribute("data-status")));
    expect(statuses).toEqual(build.rows.flatMap((row) => build.games.flatMap((game) => row.cells.filter((cell) => cell.game === game.game).map((cell) => cell.status))));
  }
  await expect(page.locator("tr[data-paired]")).toHaveCount(scorecard.paired.length);
});

test("every model is listed with its verdict and report, and a pending one with no number", async ({ page }) => {
  const results = (await (await page.request.get("data/results.json")).json()) as Results;
  await page.goto("results/");
  for (const lane of results.lanes) {
    const row = page.locator(`tr[data-lane="${lane.lane}"]`);
    await expect(row).toHaveAttribute("data-verdict", lane.verdict_class);
    await expect(row).toContainText(copy.results.models.verdictClass[lane.verdict_class]);
    await expect(row.locator(`[data-field="lanes[lane=${lane.lane}].verdict"]`)).toHaveText(lane.verdict);
    const heldout = row.locator("td[data-heldout]");
    if (lane.heldout) {
      const interval = !lane.heldout.point_estimate && lane.heldout.lo != null && lane.heldout.hi != null;
      if (lane.heldout.value != null && !interval) await expect(heldout).toContainText(copy.site.estimate.point);
    } else {
      await expect(heldout).toHaveText(copy.results.models.noHeldout);
    }
    const url = repoFileUrl(lane.report);
    if (url) await expect(row.locator(`a[data-report="${lane.report}"]`).first()).toHaveAttribute("href", url);
  }
  const pending = page.locator("[data-pending]");
  await expect(pending).toHaveCount(results.pending.length);
  for (const [index, lane] of results.pending.entries()) {
    await expect(pending.nth(index)).toContainText(lane.reason);
  }
});
