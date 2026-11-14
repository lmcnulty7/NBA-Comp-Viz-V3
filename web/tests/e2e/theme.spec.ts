import { expect, test, type Page } from "@playwright/test";
import { courtCase, tokenChannels, tokenRgb } from "./site";

async function pageColours(page: Page): Promise<{ background: string; ink: string }> {
  return page.evaluate(() => {
    const style = getComputedStyle(document.body);
    return { background: style.backgroundColor, ink: style.color };
  });
}

async function courtCorner(page: Page): Promise<number[]> {
  return page.locator('canvas[data-court][data-ready="true"]').evaluate((element) => {
    const context = (element as HTMLCanvasElement).getContext("2d")!;
    return Array.from(context.getImageData(2, 2, 1, 1).data.slice(0, 3));
  });
}

/** Two colours are the same when no channel differs by more than a rounding step of the canvas. */
function sameColour(actual: number[], expected: number[]): boolean {
  return actual.length === expected.length && actual.every((channel, index) => Math.abs(channel - expected[index]!) <= 2);
}

async function openCourt(page: Page): Promise<void> {
  const target = courtCase();
  await page.goto(`review/${target.gameSlug}/${target.possession}/`);
  await expect(page.locator('canvas[data-court][data-ready="true"]')).toBeVisible();
  await expect(page.getByTestId("possession-stage")).toHaveAttribute("aria-busy", "false");
}

test.describe("a reader whose system asks for light", () => {
  test.use({ colorScheme: "light" });

  test("the page and the court use the light tokens", async ({ page }) => {
    await openCourt(page);
    expect(await pageColours(page)).toEqual({
      background: await tokenRgb("light", "--ground"),
      ink: await tokenRgb("light", "--ink"),
    });
    expect(sameColour(await courtCorner(page), await tokenChannels("light", "--ground"))).toBe(true);
  });

  test('data-theme="dark" switches the page and redraws the court', async ({ page }) => {
    await openCourt(page);
    await page.evaluate(() => document.documentElement.setAttribute("data-theme", "dark"));
    expect((await pageColours(page)).background).toBe(await tokenRgb("dark", "--ground"));
    const dark = await tokenChannels("dark", "--ground");
    await expect.poll(async () => sameColour(await courtCorner(page), dark)).toBe(true);
  });
});

test.describe("a reader whose system asks for dark", () => {
  test.use({ colorScheme: "dark" });

  test("the page and the court use the dark tokens", async ({ page }) => {
    await openCourt(page);
    expect(await pageColours(page)).toEqual({
      background: await tokenRgb("dark", "--ground"),
      ink: await tokenRgb("dark", "--ink"),
    });
    expect(sameColour(await courtCorner(page), await tokenChannels("dark", "--ground"))).toBe(true);
  });

  test('data-theme="light" keeps the page light', async ({ page }) => {
    await openCourt(page);
    await page.evaluate(() => document.documentElement.setAttribute("data-theme", "light"));
    expect((await pageColours(page)).background).toBe(await tokenRgb("light", "--ground"));
    const light = await tokenChannels("light", "--ground");
    await expect.poll(async () => sameColour(await courtCorner(page), light)).toBe(true);
  });
});
