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
  ["a token the script does not list", { "talk/first.md": "# Title\n\nIt moved {swap.xfg.mean_abs_diff2} points.\n" }, "placeholder {swap.xfg.mean_abs_diff2} must be {letters} or a token"],
  ["ghosts called the best defense", { "review.json": JSON.stringify({ a: "Where the best defense would stand." }) }, "an average defense"],
  ["ghosts called optimal positions", { "review.json": JSON.stringify({ a: "The optimal positions for the defenders." }) }, "an average defense"],
  ["a claim that no model saw the games", { "review.json": JSON.stringify({ games: { heldout: "No model saw them while it was fitted or tuned." } }) }, "review.json.games.heldout: forbidden claim"],
  ["a claim of what a defender should do", { "results.json": JSON.stringify({ a: "Where the defender should stand." }) }, "results.json.a: forbidden claim"],
  ["a claim on a slide", { "talk/first.md": "# Title\n\nThis proves the tracker works.\n" }, "talk/first.md:3: forbidden claim"],
  ["a file that is not JSON or Markdown", { "home.txt": "words" }, "copy files are .json or .md"],
  ["broken JSON", { "home.json": "{" }, "not valid JSON"],
] as const) {
  test(`the copy check refuses ${name}`, () => {
    const done = copyCheck(files);
    expect(done.output).toContain(message);
    expect(done.status).toBe(1);
  });
}

test("presenter notes may name a word the copy never uses, to forbid it; the slide's own lines may not", () => {
  const notes = '# Title\n\nThe row is met on the point estimate.\n\n## Notes\n\n- I never say "passes".\n';
  const good = copyCheck({ "talk/first.md": notes });
  expect(good.status, good.output).toBe(0);
  const bad = copyCheck({ "talk/first.md": notes.replace("is met on the point estimate", "passes") });
  expect(bad.output).toContain("talk/first.md:3: forbidden claim");
  expect(bad.status).toBe(1);
});

test("a token of docs/SCRIPT.md may name a field with a digit; a typed number still fails", () => {
  const good = copyCheck({
    "home.json": JSON.stringify({ a: "Most gaps are under {scorecard.adopted.heldout_game.position_error.p90_ft} feet." }),
    "talk/first.md": "# Title\n\nThe typical gap is {frames.scene.p50_error_ft} feet.\n",
  });
  expect(good.status, good.output).toBe(0);
  const bad = copyCheck({ "home.json": JSON.stringify({ a: "Under {scorecard.adopted.heldout_game.position_error.p90_ft} feet, or 5." }) });
  expect(bad.output).toContain("a digit in copy");
  expect(bad.status).toBe(1);
});

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
  // The slides of PLAN.md M10 are the site's private-only routes.
  const site = privateRoutes(path.join(WEB_DIR, "app")).map((segments) => segments.join("/")).sort();
  expect(site).toEqual(["talk", "talk/[slide]"]);
});

test("a private route with no fixed folder is refused, and a leaked one is found in out/", async () => {
  const app = tempDir("app");
  for (const file of ["page.private.tsx", "[slug]/page.private.tsx", "talk/[slide]/page.private.tsx", "notes/page.private.tsx"]) {
    mkdirSync(path.dirname(path.join(app, file)), { recursive: true });
    writeFileSync(path.join(app, file), "export default function Page() { return null; }\n");
  }
  const build = (await import("../../scripts/build.mjs")) as unknown as {
    privateRoutes(appDir: string): string[][];
    uncheckablePrivateRoutes(routes: string[][]): string[][];
    leakedPrivateRoutes(outDir: string, routes: string[][]): string[][];
  };
  const routes = build.privateRoutes(app);
  const names = (list: string[][]) => list.map((segments) => `/${segments.join("/")}`).sort();
  // The home page and a top-level [param] would sit beside the public pages: no folder of out/ tells them apart.
  expect(names(build.uncheckablePrivateRoutes(routes))).toEqual(["/", "/[slug]"]);
  const out = outDir({ "index.html": "<p>home</p>", "talk/index.html": "<p>slides</p>" });
  expect(names(build.leakedPrivateRoutes(out, routes))).toEqual(["/talk/[slide]"]);
  // The site itself has none that cannot be checked.
  expect(build.uncheckablePrivateRoutes(build.privateRoutes(path.join(WEB_DIR, "app")))).toEqual([]);
});

// ------------------------------------------------------------------------------- no SportVU frame data (D60)

/** A possession file as the public data root writes it: every per-frame part empty. */
function publicPossession(extra: Record<string, unknown> = {}): string {
  return JSON.stringify({
    possession: 1,
    t: [],
    game_clock_s: [],
    shot_clock_s: [],
    players: [],
    ball: { x: [], y: [], z: [] },
    epv: { t: [0, 1, 2, 3, 4, 5, 6, 7], value: [1.1, 1.2, 1.0, 0.9, 1.3, 1.1, 0.8, 0.7] },
    ghosts: null,
    shots: [{ event: 3, t: 6.2, x: 12.4, y: 30.1, location_source: "shot_chart", xfg: 0.41 }],
    ...extra,
  });
}

const PUBLIC_DEMO = JSON.stringify({ scenes: [{ frames: [{ t: 0, boxes: [{ x: 10, y: 20, w: 5, h: 9 }], broadcast: [{ x: 40.1, y: 20.2 }], truth: [], error_ft: 1.5 }] }] });

function cleanPublicSite(): Record<string, string> {
  return {
    "index.html": `<script>self.__next_f.push([1,"{\\"frames\\":[{\\"t\\":0,\\"broadcast\\":[{\\"x\\":40.1,\\"y\\":20.2}],\\"truth\\":[]}]}"])</script>`,
    "index.txt": `1:{"frames":[{"t":0,"broadcast":[{"x":40.1,"y":20.2}],"truth":[]}]}`,
    "_next/static/chunks/app.js": "const a={x:[0,1]},b=[1,2,3,4,5,6,7,8,9],c={y:[foo,bar,baz,qux,quux,corge]};",
    "data/games/g/1.json": publicPossession(),
    "data/demo/frames.json": PUBLIC_DEMO,
  };
}

async function frameCheck(): Promise<{ frameDataIn(dir: string): string[]; tracksInText(text: string): string[] }> {
  return (await import("../../scripts/check-frame-data.mjs")) as unknown as {
    frameDataIn(dir: string): string[];
    tracksInText(text: string): string[];
  };
}

test("a public site with no frame data passes the frame check", async () => {
  const { frameDataIn } = await frameCheck();
  expect(frameDataIn(outDir(cleanPublicSite()))).toEqual([]);
  const done = runScript("check-frame-data.mjs", ["--dir", outDir(cleanPublicSite())]);
  expect(done.status, done.output).toBe(0);
  expect(done.output).toContain("no SportVU frame data");
});

for (const [name, files, found] of [
  [
    "a possession file with a player track",
    { "data/games/g/1.json": publicPossession({ t: [0, 0.2], players: [{ player_id: 7, x: [10.5, 11.2], y: [20.1, 20.4] }] }) },
    ["data/games/g/1.json: per-frame data in t, players", 'data/games/g/1.json: 2 track(s); first: 2 number(s) under "x"'],
  ],
  [
    "a single-frame possession with a ball position",
    { "data/games/g/1.json": publicPossession({ ball: { x: [47.2], y: [25.1], z: [3.4] } }) },
    ["data/games/g/1.json: per-frame data in ball.x, ball.y, ball.z", 'data/games/g/1.json: 3 track(s); first: 1 number(s) under "x"'],
  ],
  [
    "a possession file with ghosts",
    { "data/games/g/1.json": publicPossession({ ghosts: { label: "average defense", tracks: [{ player_id: 9, x: [null, 3.1], y: [null, 4.2] }] } }) },
    ["data/games/g/1.json: per-frame data in ghosts", "data/games/g/1.json: 3 track(s); first: 1 number(s)"],
  ],
  [
    "a shot located by the tracking at its release",
    { "data/games/g/1.json": publicPossession({ shots: [{ event: 3, t: 6.2, x: 12.4, y: 30.1, location_source: "tracking_at_release" }] }) },
    ["data/games/g/1.json: shot 3 carries x and y", "data/games/g/1.json: 1 track(s); first: a shot located by the tracking"],
  ],
  [
    "sensor positions in the demo frames",
    { "data/demo/frames.json": PUBLIC_DEMO.replace('"truth":[]', '"truth":[{"player_id":1,"x":40.9,"y":21.0}]') },
    ["data/demo/frames.json: 1 frame(s) with sensor positions", "data/demo/frames.json: 1 track(s); first: a non-empty truth list"],
  ],
  [
    "sensor positions in the payload of a page",
    { "index.html": `<script>self.__next_f.push([1,"{\\"truth\\":[{\\"player_id\\":1,\\"x\\":40.9,\\"y\\":21.0}]}"])</script>` },
    ["index.html: 1 track(s); first: a non-empty truth list"],
  ],
  [
    "a track in an RSC payload",
    { "review/g/1/index.txt": `3:{"possession":{"players":[{"player_id":7,"x":[10.5,11.2,11.9],"y":[20.1,20.4,20.9]}]}}` },
    ['review/g/1/index.txt: 2 track(s); first: 3 number(s) under "x"'],
  ],
  [
    "a track bundled into a script",
    { "_next/static/chunks/data.js": "const p={x:[10.5,11.2,11.9,12.4,13.0,13.6,14.1],y:[1,2]};" },
    ['_next/static/chunks/data.js: 1 track(s); first: 7 number(s) under "x"'],
  ],
] as const) {
  test(`the frame check fails on ${name}`, async () => {
    const { frameDataIn } = await frameCheck();
    const dir = outDir({ ...cleanPublicSite(), ...files });
    const problems = frameDataIn(dir);
    expect(problems.length, problems.join("\n")).toBe(found.length);
    found.forEach((start, index) => expect(problems[index]).toContain(start));
    const done = runScript("check-frame-data.mjs", ["--dir", dir]);
    expect(done.status).toBe(1);
    expect(done.output).toContain("hold SportVU frame data (PLAN.md D60)");
  });
}

test("the frame check reads JSON, escaped JSON and scripts, and leaves short constants of a script alone", async () => {
  const { tracksInText } = await frameCheck();
  expect(tracksInText('{"x":[]}')).toEqual([]);
  expect(tracksInText('{"x":12.5,"y":3}')).toEqual([]);
  expect(tracksInText('{"t":[0,1,2,3,4,5,6,7,8],"value":[1,2,3,4,5,6,7,8,9]}')).toEqual([]);
  expect(tracksInText("f={x:[0,1,2]}")).toEqual([]);
  expect(tracksInText('{"x":["a","b"]}')).toEqual([]);
  expect(tracksInText('{"box":[1,2,3,4,5,6,7,8]}')).toEqual([]);
  expect(tracksInText('{"x":[null,4.5]}')).toHaveLength(1);
  expect(tracksInText('{\\"y\\":[4.5,-2.25e1]}')).toHaveLength(1);
  expect(tracksInText("'z':[3.1]")).toHaveLength(1);
  expect(tracksInText('\\\\\\"truth\\\\\\":[{\\\\\\"x\\\\\\":1}]')).toHaveLength(1);
});
