import { expect, test } from "@playwright/test";
import { metricEstimate, sortPlayers } from "@/components/prep/sort";
import { copy, fill } from "@/lib/copy";
import { MINUS, formatSigned } from "@/lib/format";
import { players } from "./site";

/** Browser tests of the prep tab (PLAN.md M6, C9), on whatever data root the site was built from. */

const text = copy.prep;
const count = (value: number) => value.toLocaleString("en-US");

test("the player index searches by name and sorts by a metric, with the interval shown", async ({ page }) => {
  await page.goto("prep/");
  const rows = page.locator('[data-testid="player-index"] tbody tr');
  await expect(rows).toHaveCount(players.players.length);

  const target = players.players[0]!;
  const word = target.name.split(" ").at(-1)!;
  await page.getByTestId("player-search").fill(word.toUpperCase());
  const expected = players.players.filter((player) =>
    `${player.name} ${player.teams.join(" ")} ${player.position ?? ""}`.toLowerCase().includes(word.toLowerCase()),
  );
  await expect(rows).toHaveCount(expected.length);
  await expect(page.getByTestId("player-count")).toContainText(count(expected.length));
  await page.getByTestId("player-search").fill("");
  await expect(rows).toHaveCount(players.players.length);

  await page.getByTestId("player-sort").selectOption("shot_making");
  const order = sortPlayers(players.players, "shot_making").map((player) => player.slug);
  await expect.poll(async () => rows.evaluateAll((items) => items.map((item) => item.getAttribute("data-player")))).toEqual(order);
  // The sorted value carries its interval or says it is a point estimate; a player below the minimum says so.
  const cells = page.locator('[data-metric-cell="shot_making"]');
  for (let index = 0; index < Math.min(8, order.length); index += 1) {
    const player = players.players.find((candidate) => candidate.slug === order[index])!;
    const cell = (await cells.nth(index).innerText()).replace(/\s+/g, " ");
    if (metricEstimate(player, "shot_making")) {
      // The value's own words, not the baseline note beside it (which says point estimate as well).
      const value = (await cells.nth(index).locator(".estimate").innerText()).replace(/\s+/g, " ");
      expect(new RegExp(`\\d to [\\d${MINUS}]`).test(value) || value.includes(copy.site.estimate.point), value).toBe(true);
      // D51: the baseline's own error sits beside the value.
      if (player.shot_making.baseline_error_per_100 != null) {
        expect(cell).toContain(fill(text.list.baselineError, { value: formatSigned(player.shot_making.baseline_error_per_100, 1) }));
      }
    } else {
      expect(cell).toContain(text.list.below);
    }
  }
  await expect(page.getByTestId("sort-note")).toHaveText(text.list.baselineNote);
  await page.getByTestId("player-reverse").click();
  await expect
    .poll(async () => rows.evaluateAll((items) => items.map((item) => item.getAttribute("data-player"))))
    .toEqual(sortPlayers(players.players, "shot_making", true).map((player) => player.slug));

  // Absence risk carries its caution: it is not an injury forecast. Name order carries no note.
  await page.getByTestId("player-sort").selectOption("absence_risk");
  await expect(page.getByTestId("sort-note")).toHaveText(text.list.absenceNote);
  await page.getByTestId("player-sort").selectOption("name");
  await expect(page.getByTestId("sort-note")).toHaveCount(0);
});

test("a player page states the hidden players, each view's minimum sample, and the baseline's calibration", async ({ page }) => {
  const player = players.players[0]!;
  await page.goto(`prep/${player.slug}/`);
  await expect(page.getByTestId("hidden-count")).toContainText(count(players.n_hidden));
  await expect(page.getByTestId("hidden-count")).toContainText(count(players.n_players));
  const shownViews = [player.shot_making, player.gravity, player.rebounding, player.absence_risk].filter((view) => view.shown).length +
    (player.similar.available && player.similar.shown ? 1 : 0);
  await expect(page.getByTestId("min-sample")).toHaveCount(shownViews);

  // D51: the baseline's own error sits beside the value, and its calibration rows are on the page.
  const shots = page.locator('section[data-metric="shot_making"]');
  if (player.shot_making.shown && player.shot_making.baseline_error_per_100 != null) {
    await expect(shots).toContainText(text.player.shotMaking.baselineError);
  }
  const calibration = page.getByTestId("calibration");
  await calibration.locator("summary").click();
  await expect(calibration.locator("tbody tr")).toHaveCount(players.shot_making_baseline.calibration.length);

  // Absence risk: never called an injury forecast, and the observed share sits beside the model value.
  await expect(page.getByTestId("absence-caution")).toContainText(text.player.absence.caution);
  if (player.absence_risk.shown && player.absence_risk.observed_rate) {
    await expect(page.locator('section[data-metric="absence_risk"]')).toContainText(text.player.absence.observed);
  }
});

test("every player value shows an interval or says point estimate, and descriptive views say so", async ({ page }) => {
  for (const player of players.players.slice(0, 4)) {
    await page.goto(`prep/${player.slug}/`);
    const estimates = page.locator('[data-values="player"] .estimate');
    for (let index = 0; index < (await estimates.count()); index += 1) {
      const value = (await estimates.nth(index).innerText()).replace(/\s+/g, " ");
      expect(new RegExp(`\\d to [\\d${MINUS}]`).test(value) || value.includes(copy.site.estimate.point), value).toBe(true);
    }
    await expect(page.locator('section[data-metric="gravity"]')).toContainText(text.player.descriptive);
    await expect(page.locator('section[data-metric="similar"]')).toContainText(text.player.descriptive);
  }
});

test("the evidence toggle on a player page shows each view's sources", async ({ page }) => {
  const player = players.players[0]!;
  await page.goto(`prep/${player.slug}/`);
  const blocks = page.locator('[data-testid="evidence"]');
  await expect(blocks.first()).toBeHidden();
  await page.getByTestId("evidence-toggle").click();
  await expect(blocks.first()).toBeVisible();
  const paths = await page.locator('[data-testid="evidence"] code').allInnerTexts();
  for (const note of players.metrics) for (const source of note.sources) expect(paths).toContain(source.path);
});
