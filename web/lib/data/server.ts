/**
 * The data source of the build. Server components and generateStaticParams read the prepared data root,
 * public/data, which `npm run data:prepare` copies from SITE_DATA_DIR (or web/data) before every build. Those
 * are the same bytes the built site serves, so a page and the data it links to cannot disagree.
 *
 * This module opens files, so only server code may import it. A client component uses lib/data/browser.ts.
 */
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { DataError, StaticSource, type DataSource } from "./source";

const PREPARED_ROOT = path.join(process.cwd(), "public", "data");

async function readPrepared(relativePath: string): Promise<unknown> {
  if (!existsSync(PREPARED_ROOT)) {
    throw new DataError(
      relativePath,
      "missing",
      "public/data does not exist. Run `npm run data:prepare` first (the build scripts do it for you).",
    );
  }
  let text: string;
  try {
    text = await readFile(path.join(PREPARED_ROOT, relativePath), "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw new DataError(relativePath, "unreachable", (error as Error).message);
  }
  try {
    return JSON.parse(text) as unknown;
  } catch (error) {
    throw new DataError(relativePath, "invalid", (error as Error).message);
  }
}

const source: DataSource = new StaticSource(readPrepared);

export function getServerSource(): DataSource {
  return source;
}
