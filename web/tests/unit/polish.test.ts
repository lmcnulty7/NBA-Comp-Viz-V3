import { expect, test } from "@playwright/test";
import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { runScript, tempDir } from "../helpers/scripts";

// The polish loop's harness (PLAN.md P4, docs/POLISH_LOOP.md): the guard that keeps the checks read-only to the
// loop, the cost, the gate and the keep rule, and the parts of the score that need no browser.

interface Guard {
  ALLOWED: readonly string[];
  PROTECTED: readonly (readonly [string, string])[];
  forbiddenChanges(files: string[]): string[];
  changedFiles(base: string): string[];
  callCount(needle: string, ref?: string | null, root?: string): number;
  droppedChecks(base: string, root?: string): string[];
}

type Values = Record<string, number>;
interface Score {
  variant?: string;
  dataBuild?: string | null;
  check?: { ran: boolean; passed?: boolean; failedAt?: string | null };
  copy?: { problems: number };
  reading?: { problems: number };
  pageFailures?: string[];
  routes?: { count: number; hash: string };
  a11y?: { findings: Record<string, { impact: string; nodes: number }> };
}
interface Polish {
  WEIGHTS: Values;
  METRICS: string[];
  costOf(values: Values, reference: Values): { cost: number; terms: Values };
  gateReasons(score: Score, base?: Score | null): string[];
  decide(variants: Record<string, { gate: string[]; baseCost: number; cost: number; noise: number }>): { keep: boolean; reasons: string[] };
  routePattern(route: string): string;
  sampleRoutes(routes: string[]): { pattern: string; route: string }[];
  firstLoadAssets(outDir: string, route: string, basePath: string): { js: { files: number; raw: number; gzip: number }; css: { files: number } };
  copyGrade(dir: string): { grade: number; words: number; sentences: number; perFile: Record<string, number> };
}

const guard = async () => (await import("../../scripts/lib/polish-guard.mjs")) as unknown as Guard;
const polish = async () => (await import("../../scripts/lib/polish.mjs")) as unknown as Polish;

// ------------------------------------------------------------------------------------------------ the guard

for (const file of [
  "web/scripts/check.mjs",
  "web/scripts/check-copy.mjs",
  "web/scripts/check-reading.mjs",
  "web/scripts/check-contrast.mjs",
  "web/scripts/polish-score.mjs",
  "web/scripts/polish-loop.mjs",
  "web/scripts/lib/polish.mjs",
  "web/scripts/lib/polish-guard.mjs",
  "web/scripts/lib/readability.mjs",
  "web/scripts/lib/script-doc.mjs",
  "web/tests/unit/polish.test.ts",
  "web/tests/e2e/pages.spec.ts",
  "tests/test_public_docs.py",
  "web/eslint.config.mjs",
  "web/playwright.config.ts",
  "web/package.json",
  "web/package-lock.json",
  "web/next.config.ts",
  "web/lib/types/contract.ts",
  "web/lib/genai.ts",
  "web/lib/format.ts",
  "web/lib/tokens.ts",
  "web/lib/copy.ts",
  "web/lib/data/source.ts",
  "web/lib/data/server.ts",
  "web/lib/site.ts",
  "web/lib/talk.ts",
  "web/lib/repo.ts",
  "web/scripts/check-methods.mjs",
  "web/scripts/lib/generated-text.mjs",
  "tests/fixtures/genai_cases.json",
  "web/public/data/manifest.json",
  "docs/SCRIPT.md",
  "docs/POLISH_LOOP.md",
  "reports/polish/baseline-public.json",
  "export/schemas.py",
  "analytics/results.py",
  "PLAN.md",
  "CLAUDE.md",
  ".claude/settings.json",
]) {
  test(`a candidate may not change ${file}`, async () => {
    const { forbiddenChanges } = await guard();
    const problems = forbiddenChanges([file]);
    expect(problems).toHaveLength(1);
    expect(problems[0]).toContain(file);
  });
}

test("a candidate may change pages, components, client code, copy, styles and static files", async () => {
  const { forbiddenChanges } = await guard();
  expect(
    forbiddenChanges([
      "web/app/page.tsx",
      "web/app/styles/tokens.css",
      "web/components/review/Possession.tsx",
      "web/lib/nav.ts",
      "web/lib/media.ts",
      "web/app/methods/page.tsx",
      "web/content/review.json",
      "web/content/methods.json",
      "web/content/talk/clip.md",
      "web/public/stills/scene-1.jpg",
    ]),
  ).toEqual([]);
});

test("a path outside the site folders is refused, a rename counts as both paths", async () => {
  const { forbiddenChanges } = await guard();
  expect(forbiddenChanges(["README.md"])[0]).toContain("outside the site folders");
  // A new module under web/lib is outside the lint rule that keeps typed numbers out of code.
  expect(forbiddenChanges(["web/lib/helpers.ts"])[0]).toContain("outside the site folders");
  expect(forbiddenChanges(["web/app/x.tsx", "web/scripts/x.mjs"])).toHaveLength(1);
  expect(forbiddenChanges(["./web/scripts/check.mjs"])[0]).toContain("protected");
});

// The loop sets POLISH_BASE to its loop branch when it runs npm run check on a candidate, so the unit step of
// the check fails when the candidate's change since that branch touches a protected path.
test("the candidate's change touches no protected path (POLISH_BASE)", async () => {
  const base = process.env.POLISH_BASE ?? "";
  test.skip(!base, "POLISH_BASE is set only when the polish loop checks a candidate");
  const { changedFiles, droppedChecks, forbiddenChanges } = await guard();
  expect(forbiddenChanges(changedFiles(base))).toEqual([]);
  expect(droppedChecks(base)).toEqual([]);
});

test("the pages that show machine-written texts call the number check, so the required-call rule is alive", async () => {
  const { callCount } = await guard();
  expect(callCount("checkedText(", "HEAD")).toBeGreaterThanOrEqual(2);
});

test("a candidate that drops a call of the number check from a page is refused", async () => {
  const { droppedChecks } = await guard();
  const repo = tempDir("polish-repo");
  const run = (...args: string[]) => {
    const done = spawnSync("git", args, { cwd: repo, encoding: "utf8" });
    if (done.status !== 0) throw new Error(done.stderr);
  };
  run("init", "-q");
  mkdirSync(path.join(repo, "web", "app", "prep"), { recursive: true });
  const page = path.join(repo, "web", "app", "prep", "page.tsx");
  writeFileSync(page, "const a = checkedText(x, facts);\nconst b = checkedText(y, facts);\n");
  run("add", "-A");
  run("-c", "user.name=test", "-c", "user.email=", "commit", "-q", "-m", "base");
  expect(droppedChecks("HEAD", repo)).toEqual([]);
  writeFileSync(page, "const a = checkedText(x, facts);\nconst b = y.text;\n");
  expect(droppedChecks("HEAD", repo)).toEqual(["checkedText( calls fell from 2 to 1 (the page's number check of machine-written texts (lib/genai.ts))"]);
});

test("the guard's command line fails on a protected change and passes on none", () => {
  const none = runScript("lib/polish-guard.mjs", ["--base", "HEAD"]);
  // A worktree with uncommitted work may hold protected changes; the check is on the command's own answer.
  if (none.status === 0) expect(none.output).toContain("all inside the site folders");
  else expect(none.output).toContain("may not change");
  expect(runScript("lib/polish-guard.mjs", []).status).toBe(2);
});

// ----------------------------------------------------------------------------------- the cost and the gate

const REF: Values = {
  a11yWeighted: 0,
  firstViewChromiumMs: 370,
  firstViewWebkitMs: 42,
  firstLoadJsKb: 150,
  pageWeightKb: 900,
  readingGrade: 4.2,
};

test("the weights sum to one and the reference costs exactly one", async () => {
  const { WEIGHTS, METRICS, costOf } = await polish();
  expect(Object.values(WEIGHTS).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  expect([...METRICS].sort()).toEqual(Object.keys(REF).sort());
  expect(costOf(REF, REF).cost).toBe(1);
});

test("a smaller value costs less, a new accessibility violation costs more, a missing value throws", async () => {
  const { costOf, WEIGHTS } = await polish();
  const lighter = costOf({ ...REF, firstLoadJsKb: 135 }, REF);
  expect(lighter.cost).toBeCloseTo(1 - (WEIGHTS.firstLoadJsKb ?? 0) * 0.1, 4);
  expect(costOf({ ...REF, a11yWeighted: 3 }, REF).cost).toBeGreaterThan(1);
  expect(() => costOf({ ...REF, firstViewWebkitMs: undefined as unknown as number }, REF)).toThrow();
});

const passing: Score = {
  variant: "public",
  dataBuild: "site-a",
  check: { ran: true, passed: true, failedAt: null },
  copy: { problems: 0 },
  reading: { problems: 0 },
  pageFailures: [],
  routes: { count: 10, hash: "abc" },
  a11y: { findings: { "chromium|390|review/|region": { impact: "moderate", nodes: 2 } } },
};

test("the gate passes a candidate equal to its base", async () => {
  const { gateReasons } = await polish();
  expect(gateReasons(passing, passing)).toEqual([]);
});

for (const [name, change, reason] of [
  ["check not run", { check: { ran: false } }, "was not run"],
  ["check failed", { check: { ran: true, passed: false, failedAt: "browser" } }, "failed at browser"],
  ["copy problem", { copy: { problems: 1 } }, "check-copy.mjs"],
  ["reading problem", { reading: { problems: 2 } }, "check-reading.mjs"],
  ["page failure", { pageFailures: ["chromium 390 /review/: uncaught boom"] }, "page failure"],
  ["other data", { dataBuild: "site-b" }, "data build"],
  ["pages lost", { routes: { count: 9, hash: "def" } }, "built pages differ"],
  [
    "new violation",
    { a11y: { findings: { ...passing.a11y!.findings, "webkit|1280|prep/|image-alt": { impact: "critical", nodes: 1 } } } },
    "new accessibility violation webkit|1280|prep/|image-alt",
  ],
  ["grown violation", { a11y: { findings: { "chromium|390|review/|region": { impact: "moderate", nodes: 3 } } } }, "grew from 2 to 3"],
] as const) {
  test(`the gate rejects: ${name}`, async () => {
    const { gateReasons } = await polish();
    const reasons = gateReasons({ ...passing, ...(change as Partial<Score>) }, passing);
    expect(reasons.join("\n")).toContain(reason);
  });
}

test("a fixed violation passes the gate", async () => {
  const { gateReasons } = await polish();
  expect(gateReasons({ ...passing, a11y: { findings: {} } }, passing)).toEqual([]);
});

test("a candidate is kept only when it beats the noise and no variant gets worse", async () => {
  const { decide } = await polish();
  const ok = { gate: [] as string[], baseCost: 1, noise: 0.01 };
  expect(decide({ public: { ...ok, cost: 0.98 }, private: { ...ok, cost: 1.0 } }).keep).toBe(true);
  expect(decide({ public: { ...ok, cost: 0.995 } }).reasons).toEqual(["no variant improved by more than its noise"]);
  expect(decide({ public: { ...ok, cost: 0.95 }, private: { ...ok, cost: 1.02 } }).reasons[0]).toContain("private: the cost rose");
  expect(decide({ public: { ...ok, cost: null as unknown as number } }).reasons).toEqual(["public: no cost (the site was not scored)"]);
  const gated = decide({ public: { ...ok, cost: 0.5, gate: ["new accessibility violation x"] } });
  expect(gated.keep).toBe(false);
  expect(gated.reasons).toEqual(["public: new accessibility violation x"]);
});

// ------------------------------------------------------------------------------ the parts with no browser

test("one page per family is sampled, in natural order", async () => {
  const { routePattern, sampleRoutes } = await polish();
  expect(routePattern("")).toBe("");
  expect(routePattern("review/g/12/")).toBe("review/*/*/");
  expect(sampleRoutes(["", "review/", "review/b/", "review/a/", "review/a/10/", "review/a/2/", "prep/"])).toEqual([
    { pattern: "", route: "" },
    { pattern: "prep/", route: "prep/" },
    { pattern: "review/", route: "review/" },
    { pattern: "review/*/", route: "review/a/" },
    { pattern: "review/*/*/", route: "review/a/2/" },
  ]);
});

test("first-load JavaScript counts every script but the nomodule fallback, under the base path", async () => {
  const { firstLoadAssets } = await polish();
  const out = tempDir("polish-out");
  mkdirSync(path.join(out, "_next"), { recursive: true });
  writeFileSync(path.join(out, "_next", "a.js"), "x".repeat(1000));
  writeFileSync(path.join(out, "_next", "b.js"), "y".repeat(500));
  writeFileSync(path.join(out, "_next", "old.js"), "z".repeat(9000));
  writeFileSync(path.join(out, "_next", "s.css"), "body{}");
  writeFileSync(
    path.join(out, "index.html"),
    '<link rel="stylesheet" href="/B/_next/s.css"/><script src="/B/_next/a.js" async=""></script>' +
      '<script src="/B/_next/a.js"></script><script src="/B/_next/b.js"></script>' +
      '<script src="/B/_next/old.js" noModule=""></script><script src="https://example.com/x.js"></script><script>inline()</script>',
  );
  const assets = firstLoadAssets(out, "", "/B");
  expect(assets.js.files).toBe(2);
  expect(assets.js.raw).toBe(1500);
  expect(assets.js.gzip).toBeGreaterThan(0);
  expect(assets.css.files).toBe(1);
});

test("the copy grade weighs sentences by their words and leaves out labels and presenter notes", async () => {
  const { copyGrade } = await polish();
  const dir = tempDir("polish-content");
  mkdirSync(path.join(dir, "talk"), { recursive: true });
  writeFileSync(path.join(dir, "a.json"), JSON.stringify({ label: "Review", line: "The ball goes in." }));
  writeFileSync(path.join(dir, "talk", "s.md"), "---\nvisual: none\n---\n# Title\n\nWe see the court.\n\n## Notes\n\nExtraordinarily complicated methodological considerations.\n");
  const easy = copyGrade(dir);
  expect(easy.sentences).toBe(2);
  expect(easy.words).toBe(8);
  expect(Object.keys(easy.perFile).sort()).toEqual(["a.json", "talk/s.md"]);
  writeFileSync(path.join(dir, "a.json"), JSON.stringify({ line: "Considerable organisational complexity characterises everything." }));
  expect(copyGrade(dir).grade).toBeGreaterThan(easy.grade);
});

test("the scorer and the driver refuse to run without the heavy-job lock", () => {
  const scorer = runScript("polish-score.mjs", [], { V3_HEAVY_HELD: "" });
  expect(scorer.status).toBe(2);
  expect(scorer.output).toContain("heavy-job lock");
  const driver = runScript("polish-loop.mjs", ["candidate", "--run", "x"], { V3_HEAVY_HELD: "" });
  expect(driver.status).toBe(1);
  expect(driver.output).toContain("heavy-job lock");
});

// ------------------------------------------------------------------------- the data the reading check reads

interface Scorer {
  builtDataDir(outDir: string): string;
  staticChecks(outDir: string): { reading: { problems: string[]; generated: number } };
}
const scorer = async () => (await import("../../scripts/polish-score.mjs")) as unknown as Scorer;

test("the reading check of a score reads the machine-written texts of the scored site's own data", async () => {
  const { builtDataDir, staticChecks } = await scorer();
  const out = tempDir("polish-built");
  expect(builtDataDir(out)).toBe(path.join(out, "data"));
  const clean = staticChecks(out).reading;
  expect(clean.generated).toBe(0);
  expect(clean.problems.filter((p) => p.startsWith("genai/"))).toEqual([]);
  mkdirSync(path.join(out, "data", "genai"), { recursive: true });
  writeFileSync(
    path.join(out, "data", "genai", "scouting.json"),
    JSON.stringify({ items: [{ key: "someone", text: "Too short to be a text." }] }),
  );
  const read = staticChecks(out).reading;
  expect(read.generated).toBe(1);
  expect(read.problems.join("\n")).toContain("genai/scouting.json someone");
});

// ------------------------------------------------------------------------------- byte ranges for the clips

interface RangeServer {
  rangeAwareServer(options: { dir: string; base: string }): import("node:http").Server;
}

test("the scorer's server answers a byte range of a clip as a real host does, and leaves other requests to serve-out", async () => {
  const { rangeAwareServer } = (await import("../../scripts/polish-score.mjs")) as unknown as RangeServer;
  const out = tempDir("polish-range");
  mkdirSync(path.join(out, "clips"), { recursive: true });
  writeFileSync(path.join(out, "clips", "a.mp4"), "0123456789");
  writeFileSync(path.join(out, "index.html"), "<p>home</p>");
  const server = rangeAwareServer({ dir: out, base: "/B" });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const origin = `http://127.0.0.1:${(server.address() as import("node:net").AddressInfo).port}`;
  try {
    const part = await fetch(`${origin}/B/clips/a.mp4`, { headers: { Range: "bytes=2-5" } });
    expect(part.status).toBe(206);
    expect(part.headers.get("content-range")).toBe("bytes 2-5/10");
    expect(await part.text()).toBe("2345");
    const open = await fetch(`${origin}/B/clips/a.mp4`, { headers: { Range: "bytes=7-" } });
    expect(await open.text()).toBe("789");
    const tail = await fetch(`${origin}/B/clips/a.mp4`, { headers: { Range: "bytes=-3" } });
    expect(await tail.text()).toBe("789");
    expect((await fetch(`${origin}/B/clips/a.mp4`, { headers: { Range: "bytes=20-" } })).status).toBe(416);
    const whole = await fetch(`${origin}/B/clips/a.mp4`);
    expect(whole.status).toBe(200);
    expect(await whole.text()).toBe("0123456789");
    expect((await fetch(`${origin}/B/`)).status).toBe(200);
    expect((await fetch(`${origin}/clips/a.mp4`, { headers: { Range: "bytes=0-1" } })).status).toBe(404);
    expect((await fetch(`${origin}/B/index.html`, { headers: { Range: "bytes=0-1" } })).status).toBe(200);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("the scorer's server sends text gzipped when the browser accepts it, as the real hosts do", async () => {
  const { rangeAwareServer } = (await import("../../scripts/polish-score.mjs")) as unknown as RangeServer;
  const { gunzipSync } = await import("node:zlib");
  const { request } = await import("node:http");
  const out = tempDir("polish-gzip");
  mkdirSync(path.join(out, "review"), { recursive: true });
  mkdirSync(path.join(out, "stills"), { recursive: true });
  const page = `<p>${"possession ".repeat(500)}</p>`;
  writeFileSync(path.join(out, "review", "index.html"), page);
  writeFileSync(path.join(out, "app.js"), "console.log(1);".repeat(200));
  writeFileSync(path.join(out, "stills", "a.jpg"), "jpegbytes");
  writeFileSync(path.join(out, "404.html"), "<p>missing</p>");
  const server = rangeAwareServer({ dir: out, base: "/B" });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const port = (server.address() as import("node:net").AddressInfo).port;
  // fetch() would decode the body itself; a raw request shows what is on the wire.
  const get = (url: string, headers: Record<string, string> = {}) =>
    new Promise<{ status: number; headers: import("node:http").IncomingHttpHeaders; body: Buffer }>((resolve, reject) => {
      request({ host: "127.0.0.1", port, path: url, headers }, (response) => {
        const chunks: Buffer[] = [];
        response.on("data", (chunk: Buffer) => chunks.push(chunk));
        response.on("end", () => resolve({ status: response.statusCode ?? 0, headers: response.headers, body: Buffer.concat(chunks) }));
      })
        .on("error", reject)
        .end();
    });
  try {
    const zipped = await get("/B/review/", { "Accept-Encoding": "gzip, deflate, br" });
    expect(zipped.status).toBe(200);
    expect(zipped.headers["content-encoding"]).toBe("gzip");
    expect(Number(zipped.headers["content-length"])).toBe(zipped.body.length);
    expect(zipped.body.length).toBeLessThan(page.length / 4);
    expect(gunzipSync(zipped.body).toString()).toBe(page);
    expect((await get("/B/app.js", { "Accept-Encoding": "gzip" })).headers["content-encoding"]).toBe("gzip");
    const plain = await get("/B/review/");
    expect(plain.headers["content-encoding"]).toBeUndefined();
    expect(plain.body.toString()).toBe(page);
    expect((await get("/B/stills/a.jpg", { "Accept-Encoding": "gzip" })).headers["content-encoding"]).toBeUndefined();
    expect((await get("/B/review", { "Accept-Encoding": "gzip" })).status).toBe(301);
    const missing = await get("/B/nothing/", { "Accept-Encoding": "gzip" });
    expect(missing.status).toBe(404);
    expect((await get("/elsewhere/", { "Accept-Encoding": "gzip" })).status).toBe(404);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});
