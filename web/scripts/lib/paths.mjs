// Where things are. Every script resolves paths from here, never from the directory it was started in.
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const WEB_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
export const REPO_ROOT = path.resolve(WEB_DIR, "..");
export const OUT_DIR = path.join(WEB_DIR, "out");
export const PUBLIC_DATA_DIR = path.join(WEB_DIR, "public", "data");
export const TYPES_DIR = path.join(WEB_DIR, "lib", "types");
export const SCHEMA_FILE = path.join(TYPES_DIR, "contract.schema.json");
export const CONTRACT_TS_FILE = path.join(TYPES_DIR, "contract.ts");
export const CONTRACT_RULES_FILE = path.join(TYPES_DIR, "contract-rules.json");
export const CONTENT_DIR = path.join(WEB_DIR, "content");
export const TOKENS_FILE = path.join(WEB_DIR, "app", "styles", "tokens.css");
export const BUILD_INFO_FILE = "build-info.json";

// The base path of the GitHub Pages site. The check builds and serves the site under it.
export const PAGES_BASE_PATH = "/NBA-Comp-Viz-V3";

/** The data root the site reads: the folder named by SITE_DATA_DIR, else web/data. */
export function dataRoot(env = process.env) {
  const named = (env.SITE_DATA_DIR ?? "").trim();
  return named ? path.resolve(WEB_DIR, named) : path.join(WEB_DIR, "data");
}

/** The folder of private clips: the folder named by MEDIA_PRIVATE_DIR, else web/media-private. */
export function mediaPrivateDir(env = process.env) {
  const named = (env.MEDIA_PRIVATE_DIR ?? "").trim();
  return named ? path.resolve(WEB_DIR, named) : path.join(WEB_DIR, "media-private");
}

/** A path as a reader should see it: relative to the repository when it is inside it, else the folder name. */
export function shown(target) {
  const relative = path.relative(REPO_ROOT, target);
  if (relative && !relative.startsWith("..") && !path.isAbsolute(relative)) return relative;
  return `${path.basename(target)} (outside the repository)`;
}

export function normaliseBasePath(value) {
  const trimmed = String(value ?? "").trim().replace(/\/+$/, "");
  if (trimmed === "") return "";
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}

export function exists(target) {
  return existsSync(target);
}
