// scripts/package-cloudflare.mjs: package a built out/ for Cloudflare Pages (PLAN.md D64).
//
// Cloudflare Pages (free plan) takes at most 20,000 files per site and 25 MiB per file. Next writes five files per
// route: index.html, which holds the whole page, and four RSC payloads (.txt). Only three of the four are fetched
// when the client router navigates; the fourth, __next._full.txt, is not. This script copies out/ to a deploy
// folder without that file on the 3,162 possession routes, which brings the private site under the limit. No
// page and no navigation changes. The private browser suite runs against the packaged folder to prove it.
//
//   node scripts/package-cloudflare.mjs [--from out] [--to out-cloudflare]
//
// It fails when the result is above either limit, and prints the counts.

import { cpSync, existsSync, readdirSync, rmSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";

const MAX_FILES = 20000;
const MAX_FILE_BYTES = 25 * 1024 * 1024;

function option(argv, name, fallback) {
  const index = argv.indexOf(name);
  return index >= 0 && argv[index + 1] !== undefined ? argv[index + 1] : fallback;
}

// review/<game>/<possession>/__next._full.txt: the one payload of a possession route that no client navigation
// fetches (Chromium and WebKit fetch index.txt, __next._tree.txt and the page segment; tested with the private
// browser suite on the packaged folder).
export function isPossessionPayload(rel) {
  const parts = rel.split(sep).join("/").split("/");
  return parts.length === 4 && parts[0] === "review" && /^\d+$/.test(parts[2]) && parts[3] === "__next._full.txt";
}

function walk(dir, base = dir, out = []) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) walk(path, base, out);
    else out.push(relative(base, path));
  }
  return out;
}

export function packageSite(from, to) {
  if (!existsSync(join(from, "index.html"))) throw new Error(`${from} holds no built site (no index.html)`);
  rmSync(to, { recursive: true, force: true });
  let dropped = 0;
  cpSync(from, to, {
    recursive: true,
    filter: (source) => {
      const rel = relative(from, source);
      if (rel && isPossessionPayload(rel)) {
        dropped += 1;
        return false;
      }
      return true;
    },
  });
  const files = walk(to);
  const big = files.filter((rel) => statSync(join(to, rel)).size > MAX_FILE_BYTES);
  const bytes = files.reduce((sum, rel) => sum + statSync(join(to, rel)).size, 0);
  return { files: files.length, bytes, dropped, big };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const from = option(process.argv, "--from", "out");
  const to = option(process.argv, "--to", "out-cloudflare");
  const result = packageSite(from, to);
  console.log(`package: ${result.files} files, ${(result.bytes / 1e6).toFixed(2)} MB in ${to}; ` +
    `${result.dropped} possession payloads left out`);
  if (result.files > MAX_FILES) {
    console.error(`package: ${result.files} files is above the Cloudflare Pages limit of ${MAX_FILES}`);
    process.exit(1);
  }
  if (result.big.length) {
    console.error(`package: files above 25 MiB: ${result.big.join(", ")}`);
    process.exit(1);
  }
}
