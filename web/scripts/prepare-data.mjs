// Copy the data root into public/data, where the build reads it and from where the site serves it.
//
// The data root is the folder named by SITE_DATA_DIR, else web/data. It is written by
// `python3 -m export.site_data` (real data) or `python3 -m export.sample --out web/data` (synthetic data).
// Only files the contract knows are copied; anything else in the folder is named and left behind.
//
//   node scripts/prepare-data.mjs
import { cpSync, existsSync, mkdirSync, rmSync, statSync } from "node:fs";
import path from "node:path";
import { loadSchema, modelFor } from "./lib/contract.mjs";
import { PUBLIC_DATA_DIR, dataRoot, shown } from "./lib/paths.mjs";
import { listFiles } from "./validate-data.mjs";

export function prepareData(env = process.env, target = PUBLIC_DATA_DIR) {
  const root = dataRoot(env);
  const named = (env.SITE_DATA_DIR ?? "").trim();
  if (!existsSync(root) || !statSync(root).isDirectory() || !existsSync(path.join(root, "manifest.json"))) {
    const where = named ? `SITE_DATA_DIR (${shown(root)})` : shown(root);
    throw new Error(
      `no site data: ${where} has no manifest.json.\n` +
        "  Real data:      python3 -m export.site_data   (from the repository root), then set SITE_DATA_DIR\n" +
        "  Synthetic data: python3 -m export.sample --out web/data   (from the repository root)",
    );
  }
  const schema = loadSchema();
  const files = listFiles(root);
  const known = files.filter((rel) => modelFor(schema, rel) !== null);
  const unknown = files.filter((rel) => modelFor(schema, rel) === null);
  rmSync(target, { recursive: true, force: true });
  let bytes = 0;
  for (const rel of known) {
    const file = path.join(target, rel);
    mkdirSync(path.dirname(file), { recursive: true });
    cpSync(path.join(root, rel), file);
    bytes += statSync(file).size;
  }
  return { root, target, files: known.length, bytes, unknown };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    // --to <dir> writes somewhere other than public/data (the tests use it).
    const flag = process.argv.indexOf("--to");
    const result = prepareData(process.env, flag > 0 ? path.resolve(process.argv[flag + 1]) : PUBLIC_DATA_DIR);
    for (const rel of result.unknown) console.warn(`data: ${rel} is not a contract file and was not copied`);
    console.log(
      `data: ${result.files} file(s), ${(result.bytes / 1e6).toFixed(2)} MB copied from ${shown(result.root)} ` +
        `to ${shown(result.target)}`,
    );
  } catch (error) {
    console.error(`data: ${error.message}`);
    process.exitCode = 1;
  }
}
