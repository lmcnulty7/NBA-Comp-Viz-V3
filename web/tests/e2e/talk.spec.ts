import { expect, test } from "@playwright/test";
import deck from "@/content/talk/deck.json";
import { showValue } from "@/lib/tokens";
import type { DemoSwap } from "@/lib/types/contract";
import { build } from "./site";
import { watch } from "./watch";

test("the public site has no slides", async ({ request }) => {
  test.skip(build.variant !== "public", "this is the private variant");
  expect((await request.get("talk/")).status()).toBe(404);
  for (const id of deck.slides) expect((await request.get(`talk/${id}/`)).status(), id).toBe(404);
});

test.describe("the slides of the private site", () => {
  test.skip(build.variant !== "private", "the slides exist only on the private site");

  test("every slide loads from this host alone, with its number", async ({ page }) => {
    await page.goto("talk/");
    for (const id of deck.slides) await expect(page.locator(`main a[href$="/talk/${id}/"]`)).toHaveCount(1);
    for (const [index, id] of deck.slides.entries()) {
      const seen = watch(page);
      const response = await page.goto(`talk/${id}/`, { waitUntil: "networkidle" });
      expect(response?.status(), id).toBe(200);
      await expect(page.locator("h1")).toHaveCount(1);
      await expect(page.getByTestId("slide-count")).toHaveText(
        deck.ui.count.replace("{n}", String(index + 1)).replace("{total}", String(deck.slides.length)),
      );
      // A clip the bundled browser cannot decode falls back to the still; that is not a failure of the site.
      expect(seen.problems.filter((problem) => !/\/clips\//.test(problem)), id).toEqual([]);
      // Offline from out/: nothing is fetched from any other host.
      expect([...seen.hosts], id).toEqual(["127.0.0.1"]);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow, `${id} scrolls sideways`).toBeLessThanOrEqual(0);
    }
  });

  test("the arrow keys move between slides and N shows the presenter notes", async ({ page }) => {
    const [first, second] = deck.slides;
    const last = deck.slides[deck.slides.length - 1]!;
    await page.goto(`talk/${first}/`);
    await expect(page.getByTestId("presenter-notes")).toBeHidden();
    await page.keyboard.press("ArrowRight");
    await expect(page).toHaveURL(new RegExp(`/talk/${second}/$`));
    await page.keyboard.press("ArrowLeft");
    await expect(page).toHaveURL(new RegExp(`/talk/${first}/$`));
    await page.keyboard.press("End");
    await expect(page).toHaveURL(new RegExp(`/talk/${last}/$`));
    await page.keyboard.press("Home");
    await expect(page).toHaveURL(new RegExp(`/talk/${first}/$`));
    await page.keyboard.press("n");
    await expect(page.getByTestId("presenter-notes")).toBeVisible();
    await page.keyboard.press("PageDown");
    await expect(page).toHaveURL(new RegExp(`/talk/${second}/$`));
    // The notes stay on from slide to slide, until N hides them.
    await expect(page.getByTestId("presenter-notes")).toBeVisible();
    await page.keyboard.press("n");
    await expect(page.getByTestId("presenter-notes")).toBeHidden();
  });

  test("a slide plays the court live", async ({ page }) => {
    await page.goto("talk/truth/");
    await expect(page.locator('canvas[data-court][data-ready="true"]').first()).toBeVisible();
    const range = page.locator('input[type="range"]').first();
    const play = page.getByRole("button", { name: /^(Play|Pause)$/ }).first();
    if ((await play.getAttribute("aria-pressed")) !== "true") await play.click();
    const first = await range.inputValue();
    await expect.poll(async () => range.inputValue(), { timeout: 8000 }).not.toBe(first);
    // The time bar works from the keyboard too, and the deck leaves its keys alone.
    await play.click();
    await range.focus();
    await page.keyboard.press("Home");
    await expect(range).toHaveValue("0");
    await expect(page).toHaveURL(/\/talk\/truth\/$/);
  });

  test("the numbers of the comparison slide are the data's", async ({ page }) => {
    const response = await page.request.get("data/demo/swap.json");
    test.skip(!response.ok(), "this data build has no demo comparison");
    const swap = (await response.json()) as DemoSwap;
    await page.goto("talk/swap/");
    await expect(page.locator("article")).toContainText(`That is ${showValue(swap.fill_share, "share as a percentage")} percent`);
    await page.keyboard.press("n");
    await expect(page.getByTestId("presenter-notes")).toContainText(`${showValue(swap.fill_share, "share as a percentage")} percent`);
  });
});
