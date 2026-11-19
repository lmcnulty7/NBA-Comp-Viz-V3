/**
 * The demo stills, on the server. A still is a file under public/ (the demo bridge writes them to
 * public/stills); a data build whose still is not there gets null, so a page never links a missing picture.
 * Only server code may import this module: it reads the disk while the site is built.
 */
import { existsSync } from "node:fs";
import path from "node:path";
import { assetUrl } from "@/lib/site";

const PUBLIC_DIR = path.join(process.cwd(), "public");

/** The URL of a still named by the data (a path under public/), or null when this build does not hold it. */
export function stillUrl(still: string | null | undefined): string | null {
  if (!still || still.includes("..") || path.isAbsolute(still)) return null;
  return existsSync(path.join(PUBLIC_DIR, still)) ? assetUrl(still) : null;
}

/**
 * The URL of the WebP beside a still (the demo bridge writes stills/<scene>.webp next to stills/<scene>.jpg, the
 * same frame in fewer bytes; D116), or null when this build does not hold one. The data names the JPEG only, so a
 * build without the WebP shows the JPEG as before.
 */
export function stillWebpUrl(still: string | null | undefined): string | null {
  if (!still || !/\.jpe?g$/i.test(still)) return null;
  return stillUrl(still.replace(/\.jpe?g$/i, ".webp"));
}
