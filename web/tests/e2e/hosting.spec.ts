import { expect, test } from "@playwright/test";
import { existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { SCHEMA_VERSION } from "@/lib/types/contract";
import { OUT_DIR, build, courtCase, manifest } from "./site";
import { watch } from "./watch";

/** The built site as a static host serves it: under its base path, from one host, with nothing missing. */

test("the site answers only under its base path", async ({ request, baseURL }) => {
  const origin = new URL(baseURL!).origin;
  const home = await request.get(`${origin}${build.basePath}/`);
  expect(home.status()).toBe(200);
  expect(home.headers()["content-type"]).toContain("text/html");
  test.skip(build.basePath === "", "this build has no base path");
  expect((await request.get(`${origin}/`)).status()).toBe(404);
  expect((await request.get(`${origin}/review/`)).status()).toBe(404);
  expect((await request.get(`${origin}/_next/`)).status()).toBe(404);
  // A host sends the folder without its slash to the folder, as GitHub Pages does.
  const bare = await request.get(`${origin}${build.basePath}`, { maxRedirects: 0 });
  expect(bare.status()).toBe(301);
  expect(bare.headers().location).toBe(`${build.basePath}/`);
});

test("every link, script, style and font of a page is under the base path", async ({ page }) => {
  const seen = watch(page);
  const requested: string[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.protocol.startsWith("http")) requested.push(url.pathname);
  });
  await page.goto("./", { waitUntil: "networkidle" });
  const prefix = `${build.basePath}/`;
  for (const path of requested) expect(path.startsWith(prefix), `${path} is outside the base path`).toBe(true);
  const hrefs = await page.locator("a[href], link[href]").evaluateAll((nodes) =>
    nodes.map((node) => node.getAttribute("href") ?? "").filter((href) => !href.startsWith("#")),
  );
  expect(hrefs.length).toBeGreaterThan(5);
  for (const href of hrefs) expect(href.startsWith(prefix), `${href} is outside the base path`).toBe(true);
  // The three faces are served by the site itself, so the pages work with no other host.
  const fonts = requested.filter((path) => path.endsWith(".woff2"));
  expect(fonts.length).toBeGreaterThanOrEqual(2);
  for (const font of fonts) expect(font.startsWith(`${prefix}_next/static/media/`)).toBe(true);
  expect(seen.problems).toEqual([]);
  expect([...seen.hosts]).toEqual(["127.0.0.1"]);
});

test("the three faces of the design load, and the faces a page does not use are not fetched", async ({ page }) => {
  const fonts: string[] = [];
  page.on("request", (request) => {
    if (request.url().endsWith(".woff2")) fonts.push(new URL(request.url()).pathname.split("/").pop() ?? "");
  });
  const target = courtCase();
  await page.goto(`review/${target.gameSlug}/${target.possession}/`, { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  const loaded = await page.evaluate(() =>
    [...document.fonts].filter((face) => face.status === "loaded").map((face) => face.family.replace(/["']/g, "")),
  );
  for (const family of ["Newsreader", "Libre Franklin", "IBM Plex Mono"]) expect(loaded).toContain(family);
  // Text is set in those faces: the heading in the serif, a label in the sans, a number in the mono.
  const used = await page.evaluate(() => {
    const first = (selector: string) => getComputedStyle(document.querySelector(selector) as Element).fontFamily.split(",")[0]!.replace(/["']/g, "").trim();
    return { heading: first("h1"), label: first("dt"), number: first(".num") };
  });
  expect(used).toEqual({ heading: "Newsreader", label: "Libre Franklin", number: "IBM Plex Mono" });
  // The serif italic and the medium mono are on no element of this page, so they are never requested.
  expect(fonts.some((name) => name.includes("italic"))).toBe(false);
  expect(fonts.some((name) => name.includes("_500_"))).toBe(false);
  expect(fonts.some((name) => name.includes("_400_"))).toBe(true);
});

test("the data root is served beside the pages, with the schema version the site was built for", async ({ request }) => {
  const served = await request.get("data/manifest.json");
  expect(served.status()).toBe(200);
  const document = (await served.json()) as { schema_version: string; build_id: string; synthetic: boolean };
  expect(document.schema_version).toBe(SCHEMA_VERSION);
  expect(document.build_id).toBe(manifest.build_id);
  expect(document.synthetic).toBe(build.synthetic);
  const target = courtCase();
  const possession = await request.get(`data/games/${target.gameSlug}/${target.possession}.json`);
  expect(possession.status()).toBe(200);
  expect(((await possession.json()) as { possession: number }).possession).toBe(target.possession);
  // A game folder is named by the slug, never by the dotted game id.
  expect((await request.get(`data/games/${manifest.games[0]!.game_id}/possessions.json`)).status()).toBe(404);
});

test("the public site holds no clip", async ({ request }) => {
  test.skip(build.variant !== "public", "this is the private variant");
  expect((await request.get("clips/")).status()).toBe(404);
  expect((await request.get("talk/")).status()).toBe(404);
});

test("the private site serves its clips", async ({ request }) => {
  test.skip(build.variant !== "private", "this is the public variant");
  const folder = path.join(OUT_DIR, "clips");
  const clips = existsSync(folder) ? readdirSync(folder).filter((name) => !name.startsWith(".")) : [];
  expect(clips.length, "the private build copied no clip").toBeGreaterThan(0);
  for (const clip of clips) {
    const served = await request.get(`clips/${clip}`);
    expect(served.status(), clip).toBe(200);
  }
});

test("a possession page fetches its data through the data source, once", async ({ page }) => {
  const target = courtCase();
  const data: string[] = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (path.includes("/data/")) data.push(path);
  });
  await page.goto(`review/${target.gameSlug}/${target.possession}/`, { waitUntil: "networkidle" });
  await expect(page.locator('canvas[data-court][data-ready="true"]')).toBeVisible();
  expect(data).toEqual([`${build.basePath}/data/games/${target.gameSlug}/${target.possession}.json`]);
});
