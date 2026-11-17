// Build one variant of the site into out/.
//
//   node scripts/build.mjs --variant public     no clip in out/, or the build fails
//   node scripts/build.mjs --variant private    the clips folder is copied into out/clips
//
// Steps: for the public variant, write the public data root (below); copy the data root into public/data
// (prepare-data), validate it against the contract, run `next build` (a static export), then the variant step.
// The base path comes from NEXT_PUBLIC_BASE_PATH (empty for the root of a host). The variant comes from
// --variant, else the VARIANT variable, else public.
//
// The public site carries no SportVU frame data (PLAN.md D60). The public variant reads the public data root:
// `python3 -m export.public_data` writes it from the data root (SITE_DATA_DIR, else web/data) into
// web/data-public before every public build, so the public build needs Python with pydantic (PYTHON names the
// interpreter). PUBLIC_SITE_DATA_DIR names a public root written earlier instead, and then no Python runs. After
// `next build` the public variant fails, and removes out/, when any file of out/ holds frame-level positions
// (scripts/check-frame-data.mjs). The private variant reads the full data root and runs
// `python3 -m export.schemas --validate` on it, so it needs Python with pydantic too.
//
// The private variant copies the clips folder (MEDIA_PRIVATE_DIR, else web/media-private) into out/clips and
// fails when that folder is missing. The public variant fails, and removes out/, when out/ holds a video
// file or a moving image, and when a route kept for the private site (a page.private.tsx file) is in out/. A
// private route with no fixed folder before its first [param] cannot be checked that way, so it fails both variants.
// Both variants fail, and remove out/, when a page, a script or a data file in out/ holds a local path, a
// cloud mount name or an email address: nothing private is published.
import { spawnSync } from "node:child_process";
import { cpSync, existsSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { frameDataIn } from "./check-frame-data.mjs";
import { findVideo } from "./check-no-video.mjs";
import { privateScanner } from "./lib/contract.mjs";
import {
  BUILD_INFO_FILE,
  OUT_DIR,
  PUBLIC_DATA_DIR,
  PUBLIC_ROOT_DIR,
  WEB_DIR,
  dataRoot,
  mediaPrivateDir,
  normaliseBasePath,
  shown,
} from "./lib/paths.mjs";
import { runPythonModule } from "./lib/python.mjs";
import { prepareData } from "./prepare-data.mjs";
import { validateTree } from "./validate-data.mjs";

const VARIANTS = ["public", "private"];

function option(argv, name) {
  const index = argv.indexOf(name);
  return index >= 0 ? argv[index + 1] : undefined;
}

function folderSize(dir) {
  let bytes = 0;
  let files = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      const inner = folderSize(full);
      bytes += inner.bytes;
      files += inner.files;
    } else if (entry.isFile()) {
      bytes += statSync(full).size;
      files += 1;
    }
  }
  return { bytes, files };
}

const TEXT_FILE = /\.(html|txt|js|css|json|svg|xml|map|webmanifest)$/i;

/** Every private string in the text files of a built site, as "file: string". */
export function privateStringsIn(dir) {
  const findPrivate = privateScanner();
  const hits = [];
  const walk = (folder) => {
    for (const entry of readdirSync(folder, { withFileTypes: true })) {
      const full = path.join(folder, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile() && TEXT_FILE.test(entry.name)) {
        for (const found of findPrivate(readFileSync(full, "utf8"))) {
          hits.push(`${path.relative(dir, full)}: ${JSON.stringify(found)}`);
        }
      }
    }
  };
  walk(dir);
  return hits;
}

/** Routes that exist only in the private build: every app folder that holds a page.private.tsx file. */
export function privateRoutes(appDir = path.join(WEB_DIR, "app")) {
  const routes = [];
  const walk = (dir, segments) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        // A route group "(name)" and a private folder "_name" add nothing to the URL.
        const isGroup = /^\(.*\)$/.test(entry.name);
        if (entry.name.startsWith("_")) continue;
        walk(path.join(dir, entry.name), isGroup ? segments : [...segments, entry.name]);
      } else if (/^page\.private\.tsx?$/.test(entry.name)) {
        routes.push(segments);
      }
    }
  };
  if (existsSync(appDir)) walk(appDir, []);
  return routes;
}

/** The first folder of out/ a private route would be written to (the part before its first [param]). */
export function staticPrefix(segments) {
  const prefix = [];
  for (const segment of segments) {
    if (segment.startsWith("[")) break;
    prefix.push(segment);
  }
  return prefix;
}

/**
 * Private routes the public build could not prove it left out: one with no fixed folder before its first
 * [param] (app/page.private.tsx, app/[x]/page.private.tsx) would be written beside the public pages, so no folder
 * of out/ tells it apart. Such a route is refused in both variants instead of being skipped.
 */
export function uncheckablePrivateRoutes(routes) {
  return routes.filter((segments) => staticPrefix(segments).length === 0);
}

/** The private routes a built public site holds: each one whose fixed folder is in out/. */
export function leakedPrivateRoutes(outDir, routes) {
  return routes.filter((segments) => {
    const prefix = staticPrefix(segments);
    return prefix.length > 0 && existsSync(path.join(outDir, ...prefix));
  });
}

function fail(message) {
  console.error(`build: ${message}`);
  process.exit(1);
}

/**
 * The environment whose SITE_DATA_DIR is the public data root (PLAN.md D60): PUBLIC_SITE_DATA_DIR when it is
 * set, else the folder `python3 -m export.public_data` writes from the data root. Throws when the command fails.
 */
export function publicDataEnv(env = process.env) {
  const named = (env.PUBLIC_SITE_DATA_DIR ?? "").trim();
  if (named) return { env: { ...env, SITE_DATA_DIR: path.resolve(WEB_DIR, named) }, wrote: false };
  const run = runPythonModule("export.public_data", ["--src", dataRoot(env), "--out", PUBLIC_ROOT_DIR]);
  if (run.status !== 0) {
    throw new Error(`python3 -m export.public_data failed (exit ${run.status}):\n${(run.stderr || run.stdout).trim()}`);
  }
  const summary = run.stdout.trim().split("\n").pop() ?? "";
  if (summary) console.log(`build: ${summary}`);
  return { env: { ...env, SITE_DATA_DIR: PUBLIC_ROOT_DIR }, wrote: true };
}

/**
 * The contract's own rules on a data root (`python3 -m export.schemas --validate`): the lines it prints when it
 * refuses the root, or an empty list. Throws when Python cannot run.
 */
export function contractRuleProblems(root, allowMissing = []) {
  const args = ["--validate", root];
  if (allowMissing.length) args.push("--allow-missing", ...allowMissing);
  const python = runPythonModule("export.schemas", args);
  if (python.status === 0) return [];
  const lines = `${python.stdout}${python.stderr}`.split("\n").filter(Boolean);
  return lines.length ? lines : [`exit ${python.status}`];
}

function main() {
  const argv = process.argv.slice(2);
  const variant = option(argv, "--variant") ?? process.env.VARIANT ?? "public";
  if (!VARIANTS.includes(variant)) fail(`the variant is "public" or "private", not "${variant}"`);
  const basePath = normaliseBasePath(process.env.NEXT_PUBLIC_BASE_PATH);
  const started = Date.now();

  let prepared;
  try {
    prepared = prepareData(variant === "public" ? publicDataEnv().env : process.env);
  } catch (error) {
    fail(error.message);
  }
  console.log(`build: variant ${variant}, base path "${basePath}", data from ${shown(prepared.root)}`);

  const allowMissing = (process.env.SITE_DATA_ALLOW_MISSING ?? "").split(/[\s,]+/).filter(Boolean);
  let problems;
  try {
    ({ problems } = validateTree(PUBLIC_DATA_DIR, { allowMissing }));
  } catch (error) {
    fail(error.message);
  }
  if (problems.length) {
    for (const line of problems.slice(0, 20)) console.error(`build: data: ${line}`);
    fail(`the data root does not validate against the contract (${problems.length} problem(s))`);
  }
  // The private variant reads the full data root, so no Python has run on it yet. The contract's own rules run
  // here: among them every check of a machine-written text (PLAN.md P3): its input hash, the status and lane words
  // and the rules of copy, which the page's number check (lib/genai.ts) does not repeat. The public variant had
  // them when export.public_data validated its source and its output.
  if (variant === "private") {
    let refused;
    try {
      refused = contractRuleProblems(PUBLIC_DATA_DIR, allowMissing);
    } catch (error) {
      fail(error.message);
    }
    if (refused.length) {
      for (const line of refused.slice(0, 20)) console.error(`build: data: python: ${line}`);
      fail("python3 -m export.schemas --validate refused the private data root (the contract's own rules)");
    }
    console.log("build: the private data root passes the contract's own rules (python3 -m export.schemas --validate)");
  }

  // A private route the public build cannot check is refused before the long step, in both variants.
  const secret = privateRoutes();
  const uncheckable = uncheckablePrivateRoutes(secret);
  if (uncheckable.length) {
    fail(
      `the private route(s) ${uncheckable.map((segments) => `/${segments.join("/")}`).join(", ")} have no fixed folder ` +
        "before their first [param], so the public build could not prove it left them out. Give each one a folder.",
    );
  }

  // The clips folder is checked before the long step, so a private build fails early.
  const media = mediaPrivateDir();
  if (variant === "private" && !(existsSync(media) && statSync(media).isDirectory())) {
    fail(
      `the private variant needs the clips folder, and ${shown(media)} does not exist. ` +
        "Set MEDIA_PRIVATE_DIR, or put the clips in web/media-private.",
    );
  }

  rmSync(OUT_DIR, { recursive: true, force: true });
  const next = spawnSync(process.execPath, [path.join(WEB_DIR, "node_modules", "next", "dist", "bin", "next"), "build"], {
    cwd: WEB_DIR,
    stdio: "inherit",
    env: { ...process.env, VARIANT: variant, NEXT_PUBLIC_BASE_PATH: basePath, NEXT_TELEMETRY_DISABLED: "1" },
  });
  if (next.status !== 0) fail(`next build failed (exit ${next.status})`);
  if (!existsSync(path.join(OUT_DIR, "index.html"))) fail("next build wrote no out/index.html");

  if (variant === "private") {
    const clips = path.join(OUT_DIR, "clips");
    cpSync(media, clips, { recursive: true, filter: (source) => !path.basename(source).startsWith(".") });
    const copied = existsSync(clips) ? folderSize(clips) : { bytes: 0, files: 0 };
    if (copied.files === 0) console.warn(`build: warning: ${shown(media)} holds no file, so the private site has no clip`);
    console.log(`build: ${copied.files} clip file(s), ${(copied.bytes / 1e6).toFixed(2)} MB copied to out/clips`);
    for (const segments of secret) {
      const folder = path.join(OUT_DIR, ...staticPrefix(segments));
      if (!existsSync(folder)) fail(`the private route /${segments.join("/")} is missing from out/`);
    }
  } else {
    const hits = findVideo(OUT_DIR);
    const leaked = leakedPrivateRoutes(OUT_DIR, secret);
    if (hits.length || leaked.length) {
      for (const hit of hits) console.error(`build: ${path.relative(OUT_DIR, hit.file)} is ${hit.why}`);
      for (const segments of leaked) console.error(`build: the private route /${segments.join("/")} is in out/`);
      rmSync(OUT_DIR, { recursive: true, force: true });
      fail("the public site must hold no clip and no private route. out/ was removed so it cannot be published.");
    }
    const frames = frameDataIn(OUT_DIR);
    if (frames.length) {
      for (const line of frames.slice(0, 20)) console.error(`build: frame data in out/${line}`);
      rmSync(OUT_DIR, { recursive: true, force: true });
      fail(
        `${frames.length} file(s) of out/ hold SportVU frame data, which the public site never carries (PLAN.md D60). ` +
          "out/ was removed so it cannot be published.",
      );
    }
  }

  const leaks = privateStringsIn(OUT_DIR);
  if (leaks.length) {
    for (const line of leaks.slice(0, 20)) console.error(`build: private string in out/${line}`);
    rmSync(OUT_DIR, { recursive: true, force: true });
    fail(`out/ holds ${leaks.length} private string(s). out/ was removed so it cannot be published.`);
  }

  const manifest = JSON.parse(readFileSync(path.join(PUBLIC_DATA_DIR, "manifest.json"), "utf8"));
  writeFileSync(
    path.join(OUT_DIR, BUILD_INFO_FILE),
    `${JSON.stringify({ variant, basePath, synthetic: manifest.synthetic === true, dataBuild: manifest.build_id ?? null }, null, 2)}\n`,
  );
  const size = folderSize(OUT_DIR);
  const seconds = ((Date.now() - started) / 1000).toFixed(1);
  console.log(
    `build: ${variant} site in ${shown(OUT_DIR)}: ${size.files} files, ${(size.bytes / 1e6).toFixed(2)} MB, ${seconds} s`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) main();
