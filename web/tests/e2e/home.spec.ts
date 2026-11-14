import { expect, test, type Page } from "@playwright/test";
import { copy } from "@/lib/copy";
import { showValue } from "@/lib/tokens";
import type { DemoSwap, Scorecard } from "@/lib/types/contract";
import { build } from "./site";

const open = copy.home.coldOpen;
const SCENES = ["clip", "boxes", "map", "truth", "swap", "next"] as const;

async function served<T>(page: Page, path: string): Promise<T | null> {
  const response = await page.request.get(path);
  return response.ok() ? ((await response.json()) as T) : null;
}

test("the home page opens on the headline and tells the story scene by scene", async ({ page }) => {
  await page.goto("./");
  await expect(page.locator("h1")).toHaveText(open.headline.title);
  await expect(page.locator('[data-scene="headline"]')).toContainText(open.headline.steps[0]!);
  for (const scene of SCENES) await expect(page.locator(`[data-scene="${scene}"]`)).toHaveCount(1);
  // Every step of scenes one to five is a focusable stop, in the order of the script.
  const steps = page.locator("[data-step]");
  expect(await steps.count()).toBeGreaterThanOrEqual(20);
  for (const index of [0, 5, 10]) await expect(steps.nth(index)).toHaveAttribute("tabindex", "0");
  await expect(page.locator('[data-scene="map"] [data-step]').first()).toHaveText(open.map.steps[0]!);
});

test("a keyboard reader moves through the steps and drives the picture", async ({ page }) => {
  await page.goto("./");
  const boxes = page.locator('[data-scene="boxes"]');
  await boxes.locator('[data-step="0"]').focus();
  await expect(boxes).toHaveAttribute("data-active-step", "0");
  await page.keyboard.press("Tab");
  await expect(boxes.locator('[data-step="1"]')).toBeFocused();
  await expect(boxes).toHaveAttribute("data-active-step", "1");
  await expect(boxes.getByTestId("score-bars")).toHaveAttribute("data-shown", "true");
  // The play control of the court map is a real button: reachable and pressed with the keyboard.
  const play = page.locator('[data-scene="truth"]').getByRole("button");
  if (await play.count()) {
    await play.focus();
    await page.waitForTimeout(1000);
    const before = await play.getAttribute("aria-pressed");
    await page.keyboard.press("Enter");
    await expect(play).not.toHaveAttribute("aria-pressed", before ?? "");
  }
});

test("the comparison is an illustration that states n, the mean difference and the fill share (C6)", async ({ page }) => {
  const swap = await served<DemoSwap>(page, "data/demo/swap.json");
  test.skip(!swap, "this data build has no demo comparison");
  await page.goto("./");
  const scene = page.locator('[data-scene="swap"]');
  await expect(scene.getByTestId("illustration-badge")).toHaveText(open.swap.labels.badge);
  await expect(scene).toContainText("This is an illustration, not a test.");
  await expect(scene).toContainText(`on ${showValue(swap!.n_windows, "count, as stored")} stretches`);
  const share = showValue(swap!.fill_share, "share as a percentage");
  await expect(scene).toContainText(`That is ${share} percent of all player positions.`);
  await expect(scene.getByTestId("fill-share")).toContainText(`${share}%`);
  const rules = { xfg: "probability as percentage points", epv: "points, as stored", ghosting: "feet, as stored" } as const;
  for (const summary of swap!.summaries) {
    if (summary.available && summary.mean_abs_diff != null) {
      await expect(scene).toContainText(`moved by ${showValue(summary.mean_abs_diff, rules[summary.kind])} `);
      await expect(scene).toContainText(`across ${showValue(summary.n, "count, as stored")} `);
    } else {
      await expect(scene).toContainText(open.swap.fallback);
    }
  }
});

test("the gap of the scene is on screen only beside the scorecard value of the game (D50)", async ({ page }) => {
  const scorecard = (await served<Scorecard>(page, "data/scorecard.json"))!;
  await page.goto("./");
  const panel = page.locator('[data-scene="truth"]').getByTestId("gap-panel");
  test.skip((await panel.count()) === 0, "this data build has no demo scenes");
  const adopted = scorecard.builds.find((candidate) => candidate.role === "adopted")!;
  const cell = adopted.rows
    .find((row) => row.key === "position_error")!
    .cells.find((candidate) => candidate.split === "heldout_game")!;
  await expect(panel.getByTestId("gap-readout")).toBeVisible();
  await expect(panel.getByTestId("scorecard-row")).toContainText(open.truth.labels.row);
  await expect(panel.getByTestId("scorecard-row")).toContainText(showValue(cell.values.p50_ft!, "feet, as stored"));
  const step = page.locator('[data-scene="truth"] [data-step="1"]');
  await expect(step).toContainText(`Across all the scored stretches of this game`);
});

test("the public site shows a still and the request-access note, and no clip", async ({ page }) => {
  test.skip(build.variant !== "public", "this is the private variant");
  await page.goto("./");
  await expect(page.locator("video")).toHaveCount(0);
  await expect(page.locator('[data-scene="clip"]').getByTestId("access-note")).toHaveText(open.clip.caption);
});

test("the private site plays the clip and carries no request-access note", async ({ page }) => {
  test.skip(build.variant !== "private", "this is the public variant");
  await page.goto("./");
  await expect(page.getByTestId("access-note")).toHaveCount(0);
  const clip = page.locator('[data-scene="clip"] video');
  await expect(clip).toHaveCount(1);
  expect(await clip.getAttribute("src")).toContain(`${build.basePath}/clips/`);
});

test.describe("a reader who asked for reduced motion", () => {
  test.use({ reducedMotion: "reduce" });

  test("nothing plays on its own and each picture shows its final state", async ({ page }) => {
    await page.goto("./");
    await page.locator('[data-scene="truth"]').scrollIntoViewIfNeeded();
    await page.waitForTimeout(600);
    await expect(page.locator('button[aria-pressed="true"]')).toHaveCount(0);
    const bars = page.getByTestId("score-bars");
    if (await bars.count()) await expect(bars).toHaveAttribute("data-shown", "true");
    const playing = await page.locator("video").evaluateAll((videos) => videos.filter((video) => !(video as HTMLVideoElement).paused).length);
    expect(playing).toBe(0);
  });
});

test.describe("a reader with no motion preference", () => {
  test.use({ reducedMotion: "no-preference" });

  test("the court map plays when it comes into view", async ({ page }) => {
    await page.goto("./");
    const scene = page.locator('[data-scene="truth"]');
    const range = scene.locator('input[type="range"]');
    test.skip((await range.count()) === 0, "this data build has no demo frames");
    await scene.scrollIntoViewIfNeeded();
    const first = await range.inputValue();
    await expect.poll(async () => range.inputValue(), { timeout: 8000 }).not.toBe(first);
  });
});
