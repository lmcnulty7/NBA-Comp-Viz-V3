import { expect, test } from "@playwright/test";
import { TOKENS } from "@/lib/tokens";
import { copy, fill, label } from "@/lib/copy";
import {
  MINUS,
  formatClock,
  formatCount,
  formatNumber,
  formatPercent,
  formatProbability,
  formatSeconds,
  formatSigned,
} from "@/lib/format";
import { isCurrentSection } from "@/lib/nav";
import { routes } from "@/lib/site";

test("numbers keep a fixed count of decimals and use the true minus sign", () => {
  expect(MINUS).toBe(String.fromCodePoint(0x2212));
  expect(formatNumber(1.84, 2)).toBe("1.84");
  expect(formatNumber(2.26)).toBe("2.3");
  expect(formatNumber(-0.27, 2)).toBe(`${MINUS}0.27`);
  expect(formatNumber(-0.001, 2)).toBe("0.00");
  expect(formatNumber(null)).toBe("");
  expect(formatNumber(undefined)).toBe("");
  expect(formatNumber(Number.NaN)).toBe("");
});

test("counts, signed values, shares and probabilities", () => {
  expect(formatCount(3162)).toBe("3,162");
  expect(formatCount(12)).toBe("12");
  expect(formatCount(1234567)).toBe("1,234,567");
  expect(formatCount(-1500)).toBe(`${MINUS}1,500`);
  expect(formatCount(null)).toBe("");
  expect(formatSigned(0.08, 2)).toBe("+0.08");
  expect(formatSigned(-0.19, 2)).toBe(`${MINUS}0.19`);
  expect(formatSigned(0, 2)).toBe("0.00");
  expect(formatPercent(0.818, 1)).toBe("81.8%");
  expect(formatPercent(0.25)).toBe("25%");
  expect(formatPercent(null)).toBe("");
  expect(formatProbability(0.4567)).toBe("0.457");
});

test("clocks and moments", () => {
  expect(formatClock(700)).toBe("11:40");
  expect(formatClock(59.9)).toBe("0:59");
  expect(formatClock(0)).toBe("0:00");
  expect(formatClock(-3)).toBe("0:00");
  expect(formatClock(null)).toBe("");
  expect(formatSeconds(10.64)).toBe("10.6");
});

test("fill puts values into a copy template and refuses a missing one", () => {
  expect(fill("{n} possessions from {games} games", { n: "3,162", games: 17 })).toBe("3,162 possessions from 17 games");
  expect(fill("no placeholder", {})).toBe("no placeholder");
  expect(() => fill("{n} possessions", {})).toThrow(/no value for \{n\}/);
  expect(label({ shot_made: "Made shot" }, "shot_made")).toBe("Made shot");
  expect(label({ shot_made: "Made shot" }, "new_result")).toBe("new_result");
});

test("every placeholder in the site copy is one the pages can fill", () => {
  // A template is filled where it is used; here every one is at least well formed.
  const walk = (node: unknown, path: string): void => {
    if (typeof node === "string") {
      for (const placeholder of node.match(/\{[^}]*\}/g) ?? []) {
        // A {letters} placeholder is filled by the page; a dotted token of docs/SCRIPT.md by lib/tokens.ts.
        const token = /^\{([a-z][a-z_]*(?:\.[a-z0-9_]+)+)\}$/.exec(placeholder)?.[1];
        if (token) expect(Object.keys(TOKENS), `${path}: ${node}`).toContain(token);
        else expect(placeholder, `${path}: ${node}`).toMatch(/^\{[a-zA-Z]+\}$/);
      }
      return;
    }
    for (const [key, child] of Object.entries(node as Record<string, unknown>)) walk(child, `${path}.${key}`);
  };
  walk(copy, "copy");
  expect(copy.site.fixture.title).toContain("not results");
});

test("the nav marks the section of the page, and home only for home", () => {
  expect(isCurrentSection("/", routes.home)).toBe(true);
  expect(isCurrentSection("/review/", routes.home)).toBe(false);
  expect(isCurrentSection("/review/", routes.review)).toBe(true);
  expect(isCurrentSection("/review", routes.review)).toBe(true);
  expect(isCurrentSection("/review/fixture-game-a/3/", routes.review)).toBe(true);
  expect(isCurrentSection("/reviews/", routes.review)).toBe(false);
  expect(isCurrentSection("/prep/fixture-player-1/", routes.prep)).toBe(true);
  expect(isCurrentSection("/prep/fixture-player-1/", routes.results)).toBe(false);
});

test("routes use the game slug and end in a slash", () => {
  expect(routes.game("2015-12-16-phx-at-gsw")).toBe("/review/2015-12-16-phx-at-gsw/");
  expect(routes.possession("2015-12-16-phx-at-gsw", 42)).toBe("/review/2015-12-16-phx-at-gsw/42/");
  expect(routes.player("fixture-player-1")).toBe("/prep/fixture-player-1/");
  for (const path of [routes.home, routes.review, routes.prep, routes.results]) expect(path.endsWith("/")).toBe(true);
});
