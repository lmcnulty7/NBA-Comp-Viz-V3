import { expect, test } from "@playwright/test";
import { copy } from "@/lib/copy";

async function motion(page: import("@playwright/test").Page) {
  return page.evaluate((name) => {
    const link = [...document.querySelectorAll("main a")].find((node) => node.textContent?.trim() === name) as Element;
    return {
      scroll: getComputedStyle(document.documentElement).scrollBehavior,
      transition: getComputedStyle(link).transitionDuration,
    };
  }, copy.home.coldOpen.next.labels.review);
}

test.describe("a reader who asked for reduced motion", () => {
  test.use({ reducedMotion: "reduce" });

  test("nothing glides or fades", async ({ page }) => {
    await page.goto("./");
    expect(await motion(page)).toEqual({ scroll: "auto", transition: "0s" });
  });
});

test.describe("a reader with no motion preference", () => {
  test.use({ reducedMotion: "no-preference" });

  test("links ease and the page scrolls smoothly", async ({ page }) => {
    await page.goto("./");
    const found = await motion(page);
    expect(found.scroll).toBe("smooth");
    expect(found.transition).not.toBe("0s");
  });
});
