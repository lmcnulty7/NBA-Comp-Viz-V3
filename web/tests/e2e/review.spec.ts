import { expect, test, type Page } from "@playwright/test";
import type { CourtCanvasElement } from "@/components/court/Court";
import { buildTour } from "@/components/review/tour";
import { copy } from "@/lib/copy";
import type { Possession, Results } from "@/lib/types/contract";
import { build, courtCase, manifest, possessionIndex } from "./site";
import { watch } from "./watch";

/** Browser tests of the possession review (PLAN.md M5), on whatever data root the site was built from. */

const text = copy.review;

async function openStage(page: Page, gameSlug: string, possession: number, query = "") {
  await page.goto(`review/${gameSlug}/${possession}/${query}`);
  await expect(page.locator('canvas[data-court][data-ready="true"]')).toBeVisible();
  await expect(page.getByTestId("possession-stage")).toHaveAttribute("aria-busy", "false");
}

async function draws(page: Page): Promise<number> {
  return page.locator("canvas[data-court]").evaluate((element) => (element as CourtCanvasElement).courtState?.draws ?? 0);
}

async function marks(page: Page, kind: string): Promise<number> {
  return page
    .locator("canvas[data-court]")
    .evaluate((element, wanted) => (element as CourtCanvasElement).courtState?.marks.filter((mark) => mark.kind === wanted).length ?? 0, kind);
}

async function detail(page: Page, gameSlug: string, possession: number): Promise<Possession> {
  return JSON.parse(await (await page.request.get(`data/games/${gameSlug}/${possession}.json`)).text()) as Possession;
}

test("the review covers held-out games only and says so (C8)", async ({ page }) => {
  await page.goto("review/");
  await expect(page.getByTestId("heldout-note")).toContainText(String(manifest.games.length));
  const splits = await page.locator("tr[data-split]").evaluateAll((rows) => rows.map((row) => row.getAttribute("data-split")));
  expect(splits).toHaveLength(manifest.games.length);
  for (const split of splits) expect(["heldout_game", "heldout_arena"]).toContain(split);
  const target = courtCase();
  await page.goto(`review/${target.gameSlug}/${target.possession}/`);
  await expect(page.getByTestId("heldout-flag")).toBeVisible();
});

test("the possession list filters by result, xFG fifth and featured, and keeps the filter in the address", async ({ page }) => {
  const game = manifest.games[0]!;
  const index = possessionIndex(game.slug);
  await page.goto(`review/${game.slug}/`);
  const rows = page.locator('[data-testid="possession-list"] tbody tr');
  await expect(rows).toHaveCount(index.possessions.length);

  await page.getByTestId("filter-featured").check();
  const featured = index.possessions.filter((row) => row.featured);
  await expect(rows).toHaveCount(featured.length);
  await expect(page).toHaveURL(/featured=1/);
  await page.getByTestId("filter-featured").uncheck();

  const result = index.possessions[0]!.result;
  await page.getByTestId("filter-result").selectOption(result);
  await expect(rows).toHaveCount(index.possessions.filter((row) => row.result === result).length);
  for (const value of await rows.evaluateAll((items) => items.map((item) => item.getAttribute("data-result")))) {
    expect(value).toBe(result);
  }

  await page.getByTestId("filter-result").selectOption("all");
  const fifth = index.possessions.find((row) => row.xfg_bucket != null)?.xfg_bucket;
  if (fifth != null) {
    await page.getByTestId("filter-fifth").selectOption(String(fifth));
    await expect(rows).toHaveCount(index.possessions.filter((row) => row.xfg_bucket === fifth).length);
    await expect(page.getByTestId("filter-count")).toContainText(String(index.possessions.filter((row) => row.xfg_bucket === fifth).length));
    // The filter is in the address, so a reload keeps it.
    await page.reload();
    await expect(page.getByTestId("filter-fifth")).toHaveValue(String(fifth));
    await expect(rows).toHaveCount(index.possessions.filter((row) => row.xfg_bucket === fifth).length);
  }
});

test("play, pause, speed and scrub move the court and the EPV cursor together", async ({ page }) => {
  const target = courtCase();
  const seen = watch(page);
  await openStage(page, target.gameSlug, target.possession, "?frame=0");
  const stage = page.getByTestId("possession-stage");
  const play = page.getByTestId("play-toggle");
  await expect(play).toHaveText(text.possession.playback.play);
  const chart = page.getByTestId("epv-chart");
  const hasEpv = (await chart.count()) > 0;
  const cursorAt = async () => Number(await page.locator("[data-cursor-t]").getAttribute("data-cursor-t"));

  await page.getByTestId("speed").selectOption("double");
  await play.click();
  await expect(stage).toHaveAttribute("data-playing", "true");
  await expect(play).toHaveText(text.possession.playback.pause);
  await expect.poll(async () => page.getByTestId("frame-value").innerText()).not.toContain(
    text.possession.court.frameValue.split(",")[0]!.replace("{frame}", "1").replace("{total}", String(target.frames)),
  );
  if (hasEpv) await expect.poll(cursorAt).toBeGreaterThan(0.2);

  await play.click();
  await expect(stage).toHaveAttribute("data-playing", "false");
  const stopped = await draws(page);
  const frameText = await page.getByTestId("frame-value").innerText();
  await page.waitForTimeout(400);
  expect(await draws(page)).toBe(stopped);
  expect(await page.getByTestId("frame-value").innerText()).toBe(frameText);

  // Scrub with the slider: the EPV cursor follows.
  const slider = page.getByRole("slider", { name: text.possession.court.frame });
  await slider.focus();
  await page.keyboard.press("Home");
  await expect(slider).toHaveValue("0");
  if (hasEpv) await expect.poll(cursorAt).toBe(0);
  await page.keyboard.press("End");
  await expect(slider).toHaveValue(String(target.frames - 1));
  await expect(play).toHaveText(text.possession.playback.replay);

  // Press on the EPV chart: playback jumps there.
  if (hasEpv) {
    const svg = chart.locator("svg");
    await svg.scrollIntoViewIfNeeded();
    const box = (await svg.boundingBox())!;
    await svg.click({ position: { x: box.width * 0.5, y: box.height * 0.5 } });
    await expect.poll(async () => Number(await slider.inputValue())).toBeLessThan(target.frames - 1);
  }
  expect(seen.problems).toEqual([]);
});

test("playback draws at least 30 frames a second in headless Chromium at laptop width", async ({ page }, info) => {
  test.skip(info.project.name !== "chromium-1280", "measured once, on the laptop project");
  const target = courtCase();
  await openStage(page, target.gameSlug, target.possession, "?frame=0");
  await page.getByTestId("play-toggle").click();
  await expect(page.getByTestId("possession-stage")).toHaveAttribute("data-tween", "true");
  // Count court redraws (each one a full canvas draw from a new scene) and animation-frame gaps for 2 s of play.
  const measured = await page.evaluate(async () => {
    const canvas = document.querySelector("canvas[data-court]") as CourtCanvasElement;
    const startDraws = canvas.courtState?.draws ?? 0;
    const gaps: number[] = [];
    const start = performance.now();
    let last = start;
    await new Promise<void>((resolve) => {
      const tick = (now: number) => {
        gaps.push(now - last);
        last = now;
        if (now - start < 2000) requestAnimationFrame(tick);
        else resolve();
      };
      requestAnimationFrame(tick);
    });
    const seconds = (performance.now() - start) / 1000;
    const sorted = gaps.slice(1).sort((a, b) => a - b);
    return {
      drawsPerSecond: ((canvas.courtState?.draws ?? 0) - startDraws) / seconds,
      p95GapMs: sorted[Math.floor(sorted.length * 0.95)] ?? 0,
      medianGapMs: sorted[Math.floor(sorted.length / 2)] ?? 0,
    };
  });
  info.annotations.push({ type: "playback", description: JSON.stringify(measured) });
  console.log(`playback measurement: ${JSON.stringify(measured)}`);
  expect(measured.drawsPerSecond).toBeGreaterThanOrEqual(30);
});

test("ghost defenders sit behind a toggle labelled average defense (C10)", async ({ page }) => {
  const target = courtCase();
  await openStage(page, target.gameSlug, target.possession, `?frame=${Math.floor(target.frames / 2)}`);
  const toggle = page.getByTestId("ghost-toggle");
  const labelText = await page.locator(`label[for="${await toggle.getAttribute("id")}"]`).innerText();
  expect(labelText).toContain("average defense");
  const main = (await page.locator("main").innerText()).toLowerCase();
  expect(main).not.toMatch(/\b(best|optimal|ideal|perfect)\s+(defen[cs]e|defenders?|positions?)\b/);
  const possession = await detail(page, target.gameSlug, target.possession);
  if (!possession.ghosts) {
    await expect(toggle).toBeDisabled();
    return;
  }
  await expect(toggle).toBeChecked();
  await expect.poll(() => marks(page, "ghost")).toBeGreaterThan(0);
  await toggle.uncheck();
  await expect.poll(() => marks(page, "ghost")).toBe(0);
  await toggle.check();
  await expect.poll(() => marks(page, "ghost")).toBeGreaterThan(0);
});

test("the shot marker's xFG is a point estimate, beside the lane's held-out metric from the data (D53)", async ({ page }) => {
  const target = courtCase();
  await openStage(page, target.gameSlug, target.possession);
  const possession = await detail(page, target.gameSlug, target.possession);
  const results = JSON.parse(await (await page.request.get("data/results.json")).text()) as Results;
  const card = page.getByTestId("shot-card");
  const located = [...possession.shots].reverse().find((shot) => shot.x != null && shot.y != null);
  if (located?.xfg != null) {
    await expect(card.getByTestId("shot-xfg")).toContainText(copy.site.estimate.point);
    await expect(card.getByTestId("shot-xfg")).toContainText(located.xfg.toFixed(3));
  }
  const metric = card.getByTestId("lane-metric");
  const lane = results.lanes.find((candidate) => candidate.lane === "xfg");
  if (lane?.heldout?.value != null) {
    await expect(metric).toHaveAttribute("data-lane", "xfg");
    await expect(metric).toContainText(lane.report);
    if (lane.heldout.point_estimate) await expect(metric).toContainText(copy.site.estimate.point);
    else await expect(metric).toContainText(/ to /);
  } else {
    await expect(metric).toHaveAttribute("data-lane-missing", "true");
  }
});

test("the evidence toggle shows each value's sources, read from the data", async ({ page }) => {
  const target = courtCase();
  await openStage(page, target.gameSlug, target.possession);
  const possession = await detail(page, target.gameSlug, target.possession);
  const blocks = page.locator('[data-testid="evidence"]');
  expect(await blocks.count()).toBeGreaterThan(0);
  await expect(blocks.first()).toBeHidden();
  const toggle = page.getByTestId("evidence-toggle");
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-pressed", "true");
  await expect(blocks.first()).toBeVisible();
  const shown = await page.locator('[data-testid="evidence"] code').allInnerTexts();
  for (const source of possession.sources) expect(shown).toContain(source.path);
  const rows = await page.locator('[data-testid="evidence"] [data-rows]').evaluateAll((items) => items.map((item) => item.getAttribute("data-rows")));
  for (const value of rows) expect(["heldout", "not_split"]).toContain(value);
  await toggle.click();
  await expect(blocks.first()).toBeHidden();
});

test.describe("a reader who asked for reduced motion", () => {
  test.use({ reducedMotion: "reduce" });

  test("playback steps frame by frame without gliding", async ({ page }) => {
    const target = courtCase();
    await openStage(page, target.gameSlug, target.possession, "?frame=0");
    await expect(page.getByText(text.possession.playback.reduced)).toBeVisible();
    await page.getByTestId("play-toggle").click();
    await expect(page.getByTestId("possession-stage")).toHaveAttribute("data-playing", "true");
    await expect(page.getByTestId("possession-stage")).toHaveAttribute("data-tween", "false");
  });
});

test("the guided tour walks every featured possession and ends back at the review", async ({ page }) => {
  const games = manifest.games.map((game) => ({ slug: game.slug, label: game.label, possessions: possessionIndex(game.slug).possessions }));
  const stops = buildTour(games);
  await page.goto("review/");
  if (stops.length === 0) {
    await expect(page.getByTestId("tour-intro")).toContainText(text.tour.none);
    return;
  }
  const seen = watch(page);
  await page.getByTestId("tour-start").click();
  for (const [index, stop] of stops.entries()) {
    await expect(page).toHaveURL(new RegExp(`/review/${stop.gameSlug}/${stop.possession}/$`));
    const bar = page.getByTestId("tour-bar");
    await expect(bar).toHaveAttribute("data-stop", String(index + 1));
    await expect(bar).toHaveAttribute("data-stops", String(stops.length));
    await expect(bar).toContainText(stop.featuredBy === "hand" ? text.tour.byHand : text.tour.byRule);
    await expect(page.locator('canvas[data-court][data-ready="true"]')).toBeVisible();
    await expect(page.getByTestId("possession-stage")).toHaveAttribute("aria-busy", "false");
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow, "the page scrolls sideways").toBeLessThanOrEqual(0);
    if (index < stops.length - 1) await page.getByTestId("tour-next").click();
  }
  await page.getByTestId("tour-end").click();
  await expect(page).toHaveURL(new RegExp(`${build.basePath}/review/$`));
  expect(seen.problems).toEqual([]);
});
