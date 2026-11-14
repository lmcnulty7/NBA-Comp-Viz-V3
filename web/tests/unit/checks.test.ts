import { expect, test } from "@playwright/test";
import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import path from "node:path";
import { WEB_DIR, runScript, tempDir } from "../helpers/scripts";

// ------------------------------------------------------------------------------------------------ copy check

function contentDir(files: Record<string, string>): string {
  const dir = tempDir("content");
  for (const [name, text] of Object.entries(files)) {
    mkdirSync(path.dirname(path.join(dir, name)), { recursive: true });
    writeFileSync(path.join(dir, name), text);
  }
  return dir;
}

const copyCheck = (files: Record<string, string>) => runScript("check-copy.mjs", ["--dir", contentDir(files)]);

test("copy with words and placeholders passes", () => {
  const done = copyCheck({
    "home.json": JSON.stringify({ coverage: "{possessions} possessions from {games} games.", list: ["Review", "Prep"] }),
    "talk/first.md": "# How it was built\n\nThe court map error is {error} feet.\n",
  });
  expect(done.output).toContain("2 file(s)");
  expect(done.status).toBe(0);
});

for (const [name, files, message] of [
  ["a number typed into a sentence", { "home.json": JSON.stringify({ coverage: "3,162 possessions from 17 games." }) }, "a digit in copy"],
  ["a single digit", { "home.json": JSON.stringify({ a: { b: ["fine", "top 5 players"] } }) }, "home.json.a.b[1]: a digit in copy"],
  ["a number as a JSON value", { "home.json": JSON.stringify({ games: 17 }) }, "home.json.games: 17 is not text"],
  ["a boolean as a JSON value", { "home.json": JSON.stringify({ shown: true }) }, "is not text"],
  ["a digit in a key", { "home.json": JSON.stringify({ step1: "Start" }) }, "(key): a digit in copy"],
  ["a digit in Markdown", { "talk/first.md": "# Title\n\nWe tracked 25 windows.\n" }, "talk/first.md:3: a digit in copy"],
  ["a digit from another script", { "home.json": JSON.stringify({ a: `${String.fromCodePoint(0x0663)} games` }) }, "a digit in copy"],
  ["an em dash", { "home.json": JSON.stringify({ a: `Fast${String.fromCodePoint(0x2014)}and measured` }) }, "an em dash or an en dash"],
  ["an en dash", { "talk/first.md": `Pages one${String.fromCodePoint(0x2013)}two\n` }, "an em dash or an en dash"],
  ["a placeholder with a digit", { "home.json": JSON.stringify({ a: "{n1} games" }) }, "placeholder {n1} must be {letters}"],
  ["a placeholder with a path", { "home.json": JSON.stringify({ a: "{swap.mean} points" }) }, "placeholder {swap.mean} must be {letters}"],
  ["ghosts called the best defense", { "review.json": JSON.stringify({ a: "Where the best defense would stand." }) }, "an average defense"],
  ["ghosts called optimal positions", { "review.json": JSON.stringify({ a: "The optimal positions for the defenders." }) }, "an average defense"],
  ["a file that is not JSON or Markdown", { "home.txt": "words" }, "copy files are .json or .md"],
  ["broken JSON", { "home.json": "{" }, "not valid JSON"],
] as const) {
  test(`the copy check refuses ${name}`, () => {
    const done = copyCheck(files);
    expect(done.output).toContain(message);
    expect(done.status).toBe(1);
  });
}

test("the site copy itself passes the copy check", () => {
  const done = runScript("check-copy.mjs");
  expect(done.status, done.output).toBe(0);
});

// ----------------------------------------------------------------------------------------------- lint rules

interface LintMessage {
  ruleId: string | null;
  line: number;
  message: string;
}

/** Lint a source text as if it were a file of the site, and return the messages of the copy and data rules. */
function lint(filename: string, source: string): LintMessage[] {
  const result = spawnSync(
    path.join(WEB_DIR, "node_modules", ".bin", "eslint"),
    ["--stdin", "--stdin-filename", filename, "--format", "json"],
    { cwd: WEB_DIR, input: source, encoding: "utf8" },
  );
  const report = JSON.parse(result.stdout) as { messages: LintMessage[] }[];
  return report[0]?.messages ?? [];
}

const GOOD_COMPONENT = `
import Link from "next/link";
import { copy, fill } from "@/lib/copy";
import { formatCount } from "@/lib/format";

export function Good({ games, wide }: { games: number; wide: boolean }) {
  const columns = wide ? 3 : 2;
  return (
    <section className="grid cols-2 gap-16" style={{ gridTemplateColumns: \`repeat(\${columns}, 1fr)\`, maxWidth: 640 }}>
      <h2 id="h2-games">{copy.review.games.title}</h2>
      <p>{fill(copy.home.coverage, { possessions: formatCount(games), games, season: "x" })}</p>
      <svg viewBox="0 0 100 56" width={100} height={56} aria-hidden="true">
        <path d="M0 0L100 56" strokeWidth="1.5" />
      </svg>
      <Link href="/review/" aria-label={copy.site.nav.review}>
        {wide ? copy.site.nav.review : null}
      </Link>
      <span>{games > 0 ? formatCount(games) : copy.site.none}</span> ({copy.site.none})
    </section>
  );
}
`;

test("layout code with numbers passes the lint rules", () => {
  expect(lint("components/probe/Good.tsx", GOOD_COMPONENT)).toEqual([]);
});

const BAD_COMPONENT = `
import { copy } from "@/lib/copy";

export function Bad({ games, open }: { games: number; open: boolean }) {
  return (
    <section>
      <p>17 held-out games</p>
      <p>Possession review</p>
      <p>{"17 held-out games"}</p>
      <p>{\`\${games} held-out games\`}</p>
      <p>{open ? "Open" : copy.site.none}</p>
      <p>{open && "17 games"}</p>
      <td>{17}</td>
      <button aria-label="Close">{copy.site.none}</button>
      <img alt={\`Court at \${games}\`} src="x.png" />
      <input placeholder={open ? "Search 17 games" : copy.site.none} />
      <abbr title="expected field goal">{copy.site.none}</abbr>
    </section>
  );
}
`;

test("copy or a number typed into a component fails the lint rules", () => {
  const messages = lint("components/probe/Bad.tsx", BAD_COMPONENT).filter(
    (message) => message.ruleId === "react/jsx-no-literals" || message.ruleId === "no-restricted-syntax",
  );
  const lines = [...new Set(messages.map((message) => message.line))].sort((a, b) => a - b);
  // Every line from "17 held-out games" to the title attribute is reported, and no other.
  expect(lines).toEqual([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]);
});

test("a page or a component may not read a file or fetch", () => {
  const source = `
import { readFile } from "node:fs/promises";
import fs from "fs";

export async function load() {
  const a = await fetch("/data/manifest.json");
  const b = await window.fetch("/data/players.json");
  return [a, b, fs, readFile];
}
`;
  const inApp = lint("app/probe/load.ts", source).map((message) => message.ruleId);
  expect(inApp.filter((rule) => rule === "no-restricted-imports")).toHaveLength(2);
  expect(inApp).toContain("no-restricted-globals");
  expect(inApp).toContain("no-restricted-properties");
  // The data layer is where files and fetch belong.
  const inData = lint("lib/data/probe.ts", source).map((message) => message.ruleId);
  expect(inData.filter((rule) => rule?.startsWith("no-restricted"))).toEqual([]);
});

// ---------------------------------------------------------------------------------------------- video check

const bytes = (...parts: (string | number[] | Buffer)[]) =>
  Buffer.concat(parts.map((part) => (typeof part === "string" ? Buffer.from(part, "latin1") : Buffer.from(part))));

function outDir(files: Record<string, Buffer | string>): string {
  const dir = tempDir("out");
  for (const [name, content] of Object.entries(files)) {
    mkdirSync(path.dirname(path.join(dir, name)), { recursive: true });
    writeFileSync(path.join(dir, name), content);
  }
  return dir;
}

const PNG = bytes([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], [0, 0, 0, 13], "IHDR", [0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0]);
const GIF_FRAME = bytes([0x21, 0xf9, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00], [0x2c, 0, 0, 0, 0, 1, 0, 1, 0, 0]);
const GIF_HEAD = bytes("GIF89a", [1, 0, 1, 0, 0, 0, 0]);

test("a folder of pages, data and stills holds no video", () => {
  const dir = outDir({
    "index.html": "<html></html>",
    "data/manifest.json": "{}",
    "_next/static/chunk.js": "export {}",
    "stills/scene.png": bytes(PNG, [0, 0, 0, 0], "IDAT"),
    "stills/one-frame.gif": bytes(GIF_HEAD, GIF_FRAME, [0x3b]),
    "stills/scene.webp": bytes("RIFF", [0, 0, 0, 0], "WEBP", "VP8 "),
    "stills/scene.avif": bytes([0, 0, 0, 0x1c], "ftyp", "avif", [0, 0, 0, 0]),
    "fonts/a.woff2": bytes("wOF2", [0, 1, 0, 0]),
  });
  const done = runScript("check-no-video.mjs", [dir]);
  expect(done.output).toContain("no video file and no moving image");
  expect(done.status).toBe(0);
});

for (const [name, file, content, why] of [
  ["a clip by its extension", "clips/scene.mp4", "not really a video", "a .mp4 file"],
  ["an upper-case extension", "clips/SCENE.MOV", "x", "a .mov file"],
  ["a stream playlist", "clips/scene.m3u8", "#EXTM3U", "a .m3u8 file"],
  ["an MP4 renamed to .jpg", "stills/scene.jpg", bytes([0, 0, 0, 0x20], "ftyp", "isom", [0, 0, 2, 0]), "an MP4 or QuickTime file"],
  ["a QuickTime file with no extension", "media/scene", bytes([0, 0, 0, 0x14], "ftyp", "qt  ", [0, 0, 0, 0]), "an MP4 or QuickTime file"],
  ["a WebM renamed to .bin", "media/scene.bin", bytes([0x1a, 0x45, 0xdf, 0xa3, 0x01, 0, 0, 0]), "a WebM or Matroska file"],
  ["an AVI renamed to .dat", "media/scene.dat", bytes("RIFF", [0, 0, 0, 0], "AVI ", "LIST"), "an AVI file"],
  ["an animated GIF", "stills/loop.gif", bytes(GIF_HEAD, GIF_FRAME, GIF_FRAME, [0x3b]), "an animated GIF"],
  ["an animated WebP", "stills/loop.webp", bytes("RIFF", [0, 0, 0, 0], "WEBP", "VP8X", [0, 0, 0, 0], "ANIM"), "an animated WebP"],
  ["an animated PNG", "stills/loop.png", bytes(PNG, [0, 0, 0, 8], "acTL", [0, 0, 0, 2, 0, 0, 0, 0], "IDAT"), "an animated PNG"],
  ["an AVIF sequence", "stills/loop.avif", bytes([0, 0, 0, 0x1c], "ftyp", "avis", [0, 0, 0, 0]), "an animated AVIF"],
] as const) {
  test(`the video check finds ${name}`, () => {
    const dir = outDir({ "index.html": "<html></html>", [file]: content });
    const done = runScript("check-no-video.mjs", [dir]);
    expect(done.output).toContain(`${file} is ${why}`);
    expect(done.status).toBe(1);
  });
}

test("the video check refuses a folder that does not exist", () => {
  expect(runScript("check-no-video.mjs", [path.join(tempDir("none"), "missing")]).status).toBe(1);
});

// -------------------------------------------------------------------------------------------- contrast check

test("the contrast arithmetic is the WCAG one", async () => {
  const tokens = (await import("../../scripts/lib/tokens.mjs")) as unknown as {
    contrast(a: string, b: string): number;
    luminance(rgb: number[]): number;
    parseHex(value: string): number[];
  };
  expect(tokens.contrast("#000000", "#ffffff")).toBeCloseTo(21, 10);
  expect(tokens.contrast("#fff", "#FFFFFF")).toBe(1);
  expect(tokens.contrast("#767676", "#ffffff")).toBeCloseTo(4.54, 2);
  expect(tokens.contrast("#777777", "#ffffff")).toBeLessThan(4.5);
  expect(tokens.contrast("#ffffff", "#767676")).toBe(tokens.contrast("#767676", "#ffffff"));
  expect(tokens.luminance([255, 255, 255])).toBeCloseTo(1, 12);
  expect(tokens.luminance([0, 0, 0])).toBe(0);
  expect(tokens.parseHex("#1b7A52")).toEqual([27, 122, 82]);
  expect(() => tokens.parseHex("rgb(0 0 0)")).toThrow();
});

test("the contrast check passes on the design tokens and fails on a weaker colour", () => {
  expect(runScript("check-contrast.mjs").status).toBe(0);
  const tokens = readFileSync(path.join(WEB_DIR, "app", "styles", "tokens.css"), "utf8");
  const dir = tempDir("tokens");

  // A lighter muted grey in the light theme: secondary text drops below 4.5 to 1.
  const weak = path.join(dir, "weak.css");
  writeFileSync(weak, tokens.replace("--neutral-600: #666666;", "--neutral-600: #8a8a8a;"));
  const failed = runScript("check-contrast.mjs", ["--tokens", weak]);
  expect(failed.status).toBe(1);
  expect(failed.output).toMatch(/light: --muted on --ground is \d\.\d\d to 1, below 4\.5 to 1/);

  // The two dark blocks must say the same thing.
  const lopsided = path.join(dir, "lopsided.css");
  const at = tokens.lastIndexOf("--accent: #d858c2;");
  writeFileSync(lopsided, `${tokens.slice(0, at)}--accent: #d858c3;${tokens.slice(at + "--accent: #d858c2;".length)}`);
  const differs = runScript("check-contrast.mjs", ["--tokens", lopsided]);
  expect(differs.status).toBe(1);
  expect(differs.output).toContain("the two dark blocks of tokens.css give --accent different values");
});

// ---------------------------------------------------------------------------------------------- static server

test("the static server serves the site under its base path and nothing outside it", async () => {
  // The served folder sits inside a temporary one, with a file beside it that must never be served.
  const parent = outDir({
    "site/index.html": "<h1>home</h1>",
    "site/404.html": "<h1>not found</h1>",
    "site/review/index.html": "<h1>review</h1>",
    "site/data/manifest.json": '{"ok":true}',
    "site/_next/static/app.js": "export {}",
    "secret.txt": "outside",
  });
  const dir = path.join(parent, "site");
  const { createStaticServer } = (await import("../../scripts/serve-out.mjs")) as unknown as {
    createStaticServer(options: { dir: string; base: string }): import("node:http").Server;
  };
  const server = createStaticServer({ dir, base: "/NBA-Comp-Viz-V3" });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const origin = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const get = (url: string, init: RequestInit = {}) => fetch(`${origin}${url}`, { redirect: "manual", ...init });
  try {
    const home = await get("/NBA-Comp-Viz-V3/");
    expect([home.status, await home.text()]).toEqual([200, "<h1>home</h1>"]);
    expect(home.headers.get("content-type")).toBe("text/html; charset=utf-8");
    // Outside the base path there is nothing, as on a host that serves the site under a sub-path.
    expect((await get("/")).status).toBe(404);
    expect((await get("/review/")).status).toBe(404);
    expect((await get("/data/manifest.json")).status).toBe(404);
    // A folder without its slash is redirected, a folder with it is its index.html.
    const bare = await get("/NBA-Comp-Viz-V3");
    expect([bare.status, bare.headers.get("location")]).toEqual([301, "/NBA-Comp-Viz-V3/"]);
    const folder = await get("/NBA-Comp-Viz-V3/review");
    expect([folder.status, folder.headers.get("location")]).toEqual([301, "/NBA-Comp-Viz-V3/review/"]);
    expect(await (await get("/NBA-Comp-Viz-V3/review/")).text()).toBe("<h1>review</h1>");
    const data = await get("/NBA-Comp-Viz-V3/data/manifest.json");
    expect([data.status, data.headers.get("content-type")]).toEqual([200, "application/json; charset=utf-8"]);
    expect((await get("/NBA-Comp-Viz-V3/_next/static/app.js")).headers.get("content-type")).toContain("javascript");
    // An unknown page gets the site's own 404 page, and a path that climbs out of the folder gets nothing.
    const missing = await get("/NBA-Comp-Viz-V3/nothing/");
    expect([missing.status, await missing.text()]).toEqual([404, "<h1>not found</h1>"]);
    expect((await get("/NBA-Comp-Viz-V3/%2e%2e/secret.txt")).status).toBe(404);
    expect((await get("/NBA-Comp-Viz-V3/..%2fsecret.txt")).status).toBe(404);
    const head = await get("/NBA-Comp-Viz-V3/", { method: "HEAD" });
    expect([head.status, await head.text()]).toEqual([200, ""]);
    expect((await get("/NBA-Comp-Viz-V3/", { method: "POST" })).status).toBe(405);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

// ------------------------------------------------------------------------------------------ nothing private

test("a built site that holds a local path or an email address is found", async () => {
  // Put together at run time, so this file holds no such string itself.
  const local = ["", "Users", "someone", "project"].join("/");
  const address = ["someone", "example.com"].join("@");
  const dir = outDir({
    "index.html": "<p>clean</p>",
    "review/index.html": `<p>built in ${local}</p>`,
    "_next/static/chunk.js": `const contact = "${address}";`,
    "data/manifest.json": `{"note":"see https://example.org${local}"}`,
    "fonts/a.woff2": `binary ${local}`,
  });
  const { privateStringsIn } = (await import("../../scripts/build.mjs")) as unknown as {
    privateStringsIn(dir: string): string[];
  };
  // The page and the script are found. A web address is not a local path, and a font file is not text.
  expect(privateStringsIn(dir).sort()).toEqual([
    `_next/static/chunk.js: ${JSON.stringify(address)}`,
    `review/index.html: ${JSON.stringify(`${local.split("someone")[0]}`)}`,
  ]);
});

// --------------------------------------------------------------------------------------------- private routes

test("a page.private.tsx file names a route kept out of the public build", async () => {
  const app = tempDir("app");
  for (const file of [
    "page.tsx",
    "review/[game]/page.tsx",
    "talk/[slide]/page.private.tsx",
    "(group)/notes/page.private.tsx",
    "_parts/page.private.tsx",
    "prep/page.private.test.tsx",
  ]) {
    mkdirSync(path.dirname(path.join(app, file)), { recursive: true });
    writeFileSync(path.join(app, file), "export default function Page() { return null; }\n");
  }
  const { privateRoutes } = (await import("../../scripts/build.mjs")) as unknown as {
    privateRoutes(appDir: string): string[][];
  };
  const routes = privateRoutes(app).map((segments) => segments.join("/")).sort();
  expect(routes).toEqual(["notes", "talk/[slide]"]);
  // The site has no private-only route yet: the slides of PLAN.md M10 will be the first.
  expect(privateRoutes(path.join(WEB_DIR, "app"))).toEqual([]);
});
