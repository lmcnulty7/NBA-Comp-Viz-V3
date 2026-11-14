import { expect, test } from "@playwright/test";
import { copy } from "@/lib/copy";
import { MINUS } from "@/lib/format";
import { build, manifest, players, possessionIndex, sampledRoutes } from "./site";
import { watch } from "./watch";

const count = (value: number) => value.toLocaleString("en-US");

for (const route of sampledRoutes()) {
  test(`/${route} loads cleanly`, async ({ page }) => {
    const seen = watch(page);
    const response = await page.goto(route || "./", { waitUntil: "networkidle" });
    expect(response?.status()).toBe(200);

    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    await expect(page.locator("h1")).toHaveCount(1);
    await expect(page.locator("h1")).not.toBeEmpty();
    await expect(page.locator("main#main")).toHaveCount(1);
    const title = await page.title();
    expect(title.length).toBeGreaterThan(0);
    if (route !== "") expect(title).toContain(copy.site.name);

    // The fixture banner: on every page of a synthetic build, in view, and on no page of a real one.
    const banner = page.getByTestId("fixture-banner");
    if (build.synthetic) {
      await expect(banner).toBeInViewport();
      await expect(banner).toContainText(copy.site.fixture.title);
    } else {
      await expect(banner).toHaveCount(0);
    }

    // Nothing is wider than the window.
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow, "the page scrolls sideways").toBeLessThanOrEqual(0);

    // A court page finishes drawing.
    if (await page.locator("canvas[data-court]").count()) {
      await expect(page.locator('canvas[data-court][data-ready="true"]')).toBeVisible();
      await expect(page.getByTestId("possession-stage")).toHaveAttribute("aria-busy", "false");
    }

    expect(seen.problems).toEqual([]);
    // Everything comes from this one host: no font, script or data from anywhere else.
    expect([...seen.hosts]).toEqual(["127.0.0.1"]);
  });
}

test("the fixture banner stays in view when the page scrolls", async ({ page }) => {
  test.skip(!build.synthetic, "this data build is not synthetic");
  await page.goto("results/");
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
  await expect(page.getByTestId("fixture-banner")).toBeInViewport();
});

test("an unknown address gets the not-found page", async ({ page }) => {
  const response = await page.goto("no-such-page/");
  expect(response?.status()).toBe(404);
  await expect(page.locator("h1")).toHaveText(copy.site.notFound.title);
  if (build.synthetic) await expect(page.getByTestId("fixture-banner")).toBeVisible();
  await page.getByRole("link", { name: copy.site.notFound.home }).click();
  await expect(page.locator("h1")).toHaveText(copy.home.headline);
});

test("home shows the coverage of the data build and its three entry points", async ({ page }) => {
  await page.goto("./");
  await expect(page.locator("h1")).toHaveText(copy.home.headline);
  const coverage = page.getByTestId("coverage");
  await expect(coverage).toContainText(count(manifest.n_possessions));
  await expect(coverage).toContainText(count(manifest.n_games));
  await expect(coverage).toContainText(manifest.season);
  for (const [name, path] of [
    [copy.home.entries.review.title, "review/"],
    [copy.home.entries.prep.title, "prep/"],
    [copy.home.entries.results.title, "results/"],
  ] as const) {
    await expect(page.getByRole("main").getByRole("link", { name })).toHaveAttribute("href", `${build.basePath}/${path}`);
  }
});

test("review lists every game by its slug, and a game lists every possession", async ({ page }) => {
  await page.goto("review/");
  for (const game of manifest.games) {
    const link = page.getByRole("link", { name: game.label, exact: true });
    await expect(link).toHaveAttribute("href", `${build.basePath}/review/${game.slug}/`);
  }
  // No route is built from the dotted game id.
  const hrefs = await page.locator("main a[href]").evaluateAll((links) => links.map((link) => link.getAttribute("href") ?? ""));
  for (const href of hrefs) expect(href).toMatch(new RegExp(`^${build.basePath}/review/[a-z0-9][a-z0-9-]*/$`));

  const game = manifest.games[0]!;
  const index = possessionIndex(game.slug);
  await page.getByRole("link", { name: game.label, exact: true }).click();
  await expect(page.locator("h1")).toHaveText(game.label);
  await expect(page.locator("tbody tr")).toHaveCount(index.possessions.length);
  const first = index.possessions[0]!;
  await page.getByRole("link", { name: copy.review.game.open.replace("{n}", String(first.possession)), exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`/review/${game.slug}/${first.possession}/$`));
  await expect(page.locator("h1")).toContainText(String(first.possession));
});

test("prep states how many players are hidden and gives each shown player a page", async ({ page }) => {
  await page.goto("prep/");
  const counts = page.getByTestId("player-counts");
  await expect(counts).toContainText(count(players.n_shown));
  await expect(counts).toContainText(count(players.n_hidden));
  await expect(page.locator("tbody tr")).toHaveCount(players.players.length);
  const player = players.players[0]!;
  await page.getByRole("link", { name: player.name, exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`/prep/${player.slug}/$`));
  await expect(page.locator("h1")).toHaveText(player.name);
  await expect(page.locator("section[data-metric]")).toHaveCount(5);
});

test("a player value shows its interval or says it is a point estimate, and a hidden one shows no value", async ({ page }) => {
  for (const player of players.players.slice(0, 6)) {
    await page.goto(`prep/${player.slug}/`);
    // The player's own values, not the calibration error of the shot-making baseline shown beside them.
    const estimates = page.locator('[data-values="player"] .estimate');
    expect(await estimates.count()).toBeGreaterThan(0);
    for (let index = 0; index < (await estimates.count()); index += 1) {
      const text = (await estimates.nth(index).innerText()).replace(/\s+/g, " ");
      const hasRange = new RegExp(`\\d to [\\d${MINUS}]`).test(text);
      expect(hasRange || text.includes(copy.site.estimate.point), `"${text}" on ${player.slug}`).toBe(true);
    }
    const shots = page.locator('section[data-metric="shot_making"]');
    if (player.shot_making.shown) {
      await expect(shots.locator('[data-values="player"] .estimate')).toHaveCount(1);
    } else {
      await expect(shots.locator('[data-values="player"]')).toHaveCount(0);
      await expect(shots.locator('[data-hidden="true"]')).toBeVisible();
    }
    // The baseline behind shot-making is named on every player page, with its own calibration error.
    await expect(shots.getByTestId("baseline")).toContainText(players.shot_making_baseline.name);
    await expect(shots.locator('[data-values="baseline"] .estimate')).toHaveCount(1);
  }
});

test("results shows the scorecard and every model, the failed ones included", async ({ page }) => {
  const results = JSON.parse(await (await page.request.get("data/results.json")).text()) as {
    lanes: { lane: string; verdict_class: string; verdict: string }[];
  };
  const scorecard = JSON.parse(await (await page.request.get("data/scorecard.json")).text()) as {
    builds: { build: string; role: string; rows: unknown[] }[];
  };
  await page.goto("results/");
  await expect(page.locator("[data-build]")).toHaveCount(scorecard.builds.length);
  const adopted = scorecard.builds.find((candidate) => candidate.role === "adopted");
  if (adopted) await expect(page.locator("[data-build]").first()).toHaveAttribute("data-build", adopted.build);
  const rows = page.locator("tr[data-verdict]");
  await expect(rows).toHaveCount(results.lanes.length);
  const shown = await rows.evaluateAll((items) => items.map((item) => item.getAttribute("data-verdict")));
  expect(shown).toEqual(results.lanes.map((lane) => lane.verdict_class));
  for (const [index, lane] of results.lanes.entries()) {
    const classes = copy.results.models.verdictClass;
    await expect(rows.nth(index)).toContainText(classes[lane.verdict_class as keyof typeof classes]);
    await expect(rows.nth(index)).toContainText(lane.verdict);
  }
});
