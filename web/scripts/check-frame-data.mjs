// No SportVU frame data in the public site (PLAN.md D60): the source is marked not for redistribution.
//
// The public build reads a data root written by `python3 -m export.public_data`, which empties every per-frame
// track. This check reads the built site itself, so it holds whatever the data root or a page did. It fails when
//   - a possession file under out/data/games has a frame time, a clock, a player, a ball position or a ghost;
//   - out/data/demo/frames.json has a sensor position (a non-empty truth list);
//   - a shot located by the tracking at its release carries x or y;
//   - any text file of out/ (pages, the RSC .txt payloads, scripts, styles, data) holds a serialized position
//     track: an array of coordinate-like numbers under a key x, y or z (one number is enough under a quoted key,
//     as in JSON and in the escaped JSON of a page; MIN_TRACK numbers under a bare key, as in a script), a
//     non-empty truth list, a ghost track list, a player track, or a shot whose location came from the tracking.
//
//   node scripts/check-frame-data.mjs [--dir <out dir>]
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { OUT_DIR, shown } from "./lib/paths.mjs";

// Under a bare key (a script), an array needs this many numbers to count as a track, so a short constant array
// of a library is not taken for one.
export const MIN_TRACK = 6;
const TEXT_FILE = /\.(html|txt|js|mjs|css|json|svg|xml|map|webmanifest|rsc)$/i;
const NUMBER = /^-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i;
const SCAN_LIMIT = 200_000; // characters read after an opening bracket before the array is given up on

// A key, quoted (JSON), quoted with escapes (JSON inside a string of a page or a payload), or bare (a script).
const Q = String.raw`(?:\\*["'])`;
const COORDINATE_KEY = new RegExp(String.raw`(${Q})?\b([xyz])(${Q})?\s*:\s*\[`, "g");
const TRUTH = new RegExp(String.raw`${Q}truth${Q}\s*:\s*\[\s*\{`, "g");
const GHOST_TRACKS = new RegExp(String.raw`${Q}tracks${Q}\s*:\s*\[\s*\{\s*${Q}player_id${Q}`, "g");
const TRACKED_SHOT = new RegExp(String.raw`${Q}location_source${Q}\s*:\s*${Q}tracking_at_release${Q}`, "g");

/** The flat array that starts at `start` (just after "["): its entries, or null when it holds anything but numbers and nulls. */
function flatNumbers(text, start) {
  const end = Math.min(text.length, start + SCAN_LIMIT);
  let close = -1;
  for (let index = start; index < end; index += 1) {
    const char = text[index];
    if (char === "]") {
      close = index;
      break;
    }
    if (char === "[" || char === "{" || char === '"' || char === "'" || char === "\\") return null;
  }
  if (close < 0) return null;
  const body = text.slice(start, close).trim();
  if (body === "") return { numbers: 0, entries: 0 };
  let numbers = 0;
  const entries = body.split(",");
  for (const raw of entries) {
    const entry = raw.trim();
    if (NUMBER.test(entry)) numbers += 1;
    else if (entry !== "null") return null;
  }
  return { numbers, entries: entries.length };
}

function excerpt(text, index) {
  return text.slice(Math.max(0, index - 20), index + 60).replace(/\s+/g, " ");
}

/** Every serialized position track in one text, as a reason and an excerpt. */
export function tracksInText(text) {
  const found = [];
  for (const match of text.matchAll(COORDINATE_KEY)) {
    const quoted = Boolean(match[1] && match[3]);
    const array = flatNumbers(text, match.index + match[0].length);
    if (!array || array.numbers === 0) continue;
    if (quoted || array.numbers >= MIN_TRACK) {
      found.push(`${array.numbers} number(s) under "${match[2]}": ${excerpt(text, match.index)}`);
    }
  }
  for (const [pattern, why] of [
    [TRUTH, "a non-empty truth list (sensor positions)"],
    [GHOST_TRACKS, "a ghost track list"],
    [TRACKED_SHOT, "a shot located by the tracking at its release"],
  ]) {
    for (const match of text.matchAll(pattern)) found.push(`${why}: ${excerpt(text, match.index)}`);
  }
  return found;
}

/** What the data files of a built site hold, read as data: each possession, the demo frames, every shot. */
function dataProblems(dataDir) {
  const problems = [];
  const games = path.join(dataDir, "games");
  if (existsSync(games)) {
    for (const slug of readdirSync(games).sort()) {
      const folder = path.join(games, slug);
      if (!statSync(folder).isDirectory()) continue;
      for (const name of readdirSync(folder).sort()) {
        if (!/^\d+\.json$/.test(name)) continue;
        const rel = `data/games/${slug}/${name}`;
        let possession;
        try {
          possession = JSON.parse(readFileSync(path.join(folder, name), "utf8"));
        } catch {
          problems.push(`${rel}: not JSON`);
          continue;
        }
        const filled = ["t", "game_clock_s", "shot_clock_s", "players"].filter((key) => (possession[key] ?? []).length > 0);
        for (const key of ["x", "y", "z"]) if ((possession.ball?.[key] ?? []).length > 0) filled.push(`ball.${key}`);
        if (possession.ghosts != null) filled.push("ghosts");
        if (filled.length) problems.push(`${rel}: per-frame data in ${filled.join(", ")}`);
        for (const shot of possession.shots ?? []) {
          if (shot.location_source === "tracking_at_release" && (shot.x != null || shot.y != null)) {
            problems.push(`${rel}: shot ${shot.event} carries x and y from the tracking at its release`);
          }
        }
      }
    }
  }
  const frames = path.join(dataDir, "demo", "frames.json");
  if (existsSync(frames)) {
    const demo = JSON.parse(readFileSync(frames, "utf8"));
    const truth = (demo.scenes ?? []).reduce(
      (sum, scene) => sum + (scene.frames ?? []).filter((frame) => (frame.truth ?? []).length > 0).length,
      0,
    );
    if (truth > 0) problems.push(`data/demo/frames.json: ${truth} frame(s) with sensor positions in truth`);
  }
  return problems;
}

/** Every file of `dir` that holds frame-level positions, as "file: why" lines. Empty for a clean public site. */
export function frameDataIn(dir) {
  const problems = dataProblems(path.join(dir, "data"));
  const walk = (folder) => {
    for (const entry of readdirSync(folder, { withFileTypes: true })) {
      const full = path.join(folder, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile() && TEXT_FILE.test(entry.name)) {
        const hits = tracksInText(readFileSync(full, "utf8"));
        if (hits.length) {
          problems.push(`${path.relative(dir, full)}: ${hits.length} track(s); first: ${hits[0]}`);
        }
      }
    }
  };
  walk(dir);
  return problems;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const flag = process.argv.indexOf("--dir");
  const dir = flag > 0 ? path.resolve(process.argv[flag + 1]) : OUT_DIR;
  if (!existsSync(path.join(dir, "index.html"))) {
    console.error(`frames: ${shown(dir)} holds no built site. Build it first: npm run build:public`);
    process.exit(1);
  }
  const problems = frameDataIn(dir);
  for (const line of problems.slice(0, 30)) console.error(`frames: ${line}`);
  if (problems.length) {
    console.error(`frames: ${problems.length} file(s) of ${shown(dir)} hold SportVU frame data (PLAN.md D60)`);
    process.exit(1);
  }
  console.log(`frames: no SportVU frame data in ${shown(dir)}`);
}
