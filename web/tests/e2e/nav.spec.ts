import { expect, test, type Page } from "@playwright/test";
import { copy } from "@/lib/copy";
import { build, courtCase } from "./site";

const NAV = [
  [copy.site.nav.home, ""],
  [copy.site.nav.review, "review/"],
  [copy.site.nav.prep, "prep/"],
  [copy.site.nav.results, "results/"],
] as const;

/** Safari moves between links with Option and Tab unless a system setting says otherwise; WebKit does the same. */
function tabKey(browserName: string): string {
  return browserName === "webkit" ? "Alt+Tab" : "Tab";
}

async function focusedText(page: Page): Promise<string> {
  return page.evaluate(() => (document.activeElement?.textContent ?? "").trim());
}

test("the skip link is the first stop and moves focus to the content", async ({ page, browserName }) => {
  await page.goto("./");
  await page.keyboard.press(tabKey(browserName));
  const skip = page.getByRole("link", { name: copy.site.skipLink });
  await expect(skip).toBeFocused();
  await expect(skip).toBeInViewport();
  await page.keyboard.press("Enter");
  await expect(page.locator("main#main")).toBeFocused();
  await expect(page).toHaveURL(/#main$/);
});

test("the nav is reachable by keyboard, in order, after the site name", async ({ page, browserName }) => {
  await page.goto("./");
  const order: string[] = [];
  for (let stop = 0; stop < 6; stop += 1) {
    await page.keyboard.press(tabKey(browserName));
    order.push(await focusedText(page));
  }
  expect(order).toEqual([copy.site.skipLink, copy.site.name, ...NAV.map(([name]) => name)]);
  // The focused link shows a focus ring.
  const outline = await page.evaluate(() => getComputedStyle(document.activeElement as Element).outlineStyle);
  expect(outline).not.toBe("none");
  // Enter follows it.
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(new RegExp(`${build.basePath}/results/$`));
  await expect(page.locator("h1")).toHaveText(copy.results.title);
});

test("the nav has Home, Review, Prep and Results, and marks the current one", async ({ page }) => {
  await page.goto("./");
  const nav = page.getByRole("navigation", { name: copy.site.nav.label });
  await expect(nav.getByRole("link")).toHaveText(NAV.map(([name]) => name));
  for (const [name, path] of NAV) {
    await expect(nav.getByRole("link", { name, exact: true })).toHaveAttribute("href", `${build.basePath}/${path}`);
  }
  await expect(nav.locator('[aria-current="page"]')).toHaveText(copy.site.nav.home);

  for (const [name, path] of NAV.slice(1)) {
    await nav.getByRole("link", { name, exact: true }).click();
    await expect(page).toHaveURL(new RegExp(`${build.basePath}/${path}$`));
    await expect(nav.locator('[aria-current="page"]')).toHaveText(name);
    await expect(page.locator("h1")).toBeVisible();
  }
  await page.getByRole("link", { name: copy.site.name }).click();
  await expect(page).toHaveURL(new RegExp(`${build.basePath}/$`));
  await expect(nav.locator('[aria-current="page"]')).toHaveText(copy.site.nav.home);
});

test("a page deep in a section keeps its section marked", async ({ page }) => {
  const target = courtCase();
  await page.goto(`review/${target.gameSlug}/${target.possession}/`);
  const nav = page.getByRole("navigation", { name: copy.site.nav.label });
  await expect(nav.locator('[aria-current="page"]')).toHaveText(copy.site.nav.review);
  await expect(nav.locator('[aria-current="page"]')).toHaveCount(1);
});

test("the page has its landmarks: a header, a nav, the main content and a footer", async ({ page }) => {
  await page.goto("review/");
  await expect(page.getByRole("banner")).toHaveCount(1);
  await expect(page.getByRole("navigation", { name: copy.site.nav.label })).toHaveCount(1);
  await expect(page.getByRole("main")).toHaveCount(1);
  await expect(page.getByRole("contentinfo")).toHaveCount(1);
  await expect(page.getByRole("contentinfo")).toContainText(build.dataBuild ?? "");
});
