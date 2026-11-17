// Validate a data root against the contract.
//
// With ajv, against lib/types/contract.schema.json:
//   - every file is one the contract knows, and valid against its model;
//   - no string is a local path, a cloud mount name or an email address (the scan of find_private);
//   - a file sits at the path its content names, and the manifest lists every file with its size and hash;
//   - the fixed files all exist (only the demo files may be allowed to be missing);
//   - what the site's routes rely on: every game of the manifest has an index, every possession of an index
//     has a file and the reverse, and game and player slugs are unique and usable in a URL.
//
// With --python, `python3 -m export.schemas --validate` runs afterwards on the same folder. The cross-field
// rules of the contract live in its pydantic validators (an estimate has an interval or is a point estimate, a
// possession's tracks are aligned, a shown player value clears its minimum sample, real data reads held-out
// games only, ...), which a JSON Schema cannot carry. `npm run data:validate` and `npm run check` use --python.
//
//   node scripts/validate-data.mjs [--root <dir>] [--python] [--allow-missing demo/swap.json demo/frames.json]
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import {
  describeAjvError,
  fixedFiles,
  loadRules,
  loadSchema,
  modelFor,
  privateScanner,
  validators,
} from "./lib/contract.mjs";
import { dataRoot, shown } from "./lib/paths.mjs";
import { runPythonModule } from "./lib/python.mjs";

const SLUG = /^[a-z0-9][a-z0-9-]*$/;

/** Every file under `root` as a posix path relative to it, sorted. Files the operating system drops are left out. */
export function listFiles(root, ignoredNames = loadRules().ignored_names) {
  const out = [];
  const walk = (dir, prefix) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory()) walk(path.join(dir, entry.name), relative);
      else if (entry.isFile() && !ignoredNames.includes(entry.name)) out.push(relative);
    }
  };
  walk(root, "");
  return out.sort();
}

function sha256(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

/** The manifest must list every file on disk, exactly once, with its real size and hash. */
function manifestProblems(root, manifest, seen, schema) {
  const problems = [];
  const covered = new Set();
  for (const entry of manifest.files) {
    let members;
    if (entry.count === 1 && !entry.path.includes("<")) {
      members = [entry.path];
    } else {
      const folder = entry.path.slice(0, entry.path.lastIndexOf("/"));
      members = [...seen]
        .filter((rel) => rel.startsWith(`${folder}/`) && rel.endsWith(".json") && modelFor(schema, rel) === entry.model)
        .sort();
    }
    if (members.length !== entry.count) {
      problems.push(`manifest.json: ${entry.path} lists ${entry.count} file(s), ${members.length} on disk`);
    }
    if (members.some((rel) => !seen.has(rel))) {
      problems.push(`manifest.json: ${entry.path} is listed and not on disk`);
      continue;
    }
    const buffers = members.map((rel) => readFileSync(path.join(root, rel)));
    const hashes = buffers.map(sha256);
    const size = buffers.reduce((sum, buffer) => sum + buffer.length, 0);
    const single = members.length === 1 && entry.count === 1 && !entry.path.includes("<");
    const digest = single ? hashes[0] : sha256(Buffer.from([...hashes].sort().join("")));
    if (size !== entry.bytes || digest !== entry.sha256) {
      problems.push(`manifest.json: ${entry.path} does not match the files on disk (size or hash)`);
    }
    for (const rel of members) covered.add(rel);
  }
  for (const rel of [...seen].sort()) {
    if (!covered.has(rel) && rel !== "manifest.json") problems.push(`manifest.json: ${rel} is on disk and not listed`);
  }
  return problems;
}

/** What the routes of the site need from the data root, beyond each file being valid on its own. */
function siteProblems(documents, seen) {
  const problems = [];
  const manifest = documents.get("manifest.json");
  if (manifest) {
    const slugs = manifest.games.map((game) => game.slug);
    if (new Set(slugs).size !== slugs.length) problems.push("manifest.json: a game slug is listed twice");
    for (const slug of slugs) {
      if (!SLUG.test(slug)) problems.push(`manifest.json: game slug ${JSON.stringify(slug)} cannot be used in a URL`);
      if (!seen.has(`games/${slug}/possessions.json`)) {
        problems.push(`manifest.json: game ${slug} has no games/${slug}/possessions.json`);
      }
    }
    for (const rel of seen) {
      const parts = rel.split("/");
      if (parts[0] === "games" && parts[2] === "possessions.json" && !slugs.includes(parts[1])) {
        problems.push(`${rel}: the manifest does not list this game`);
      }
    }
  }
  for (const [rel, index] of documents) {
    if (!rel.startsWith("games/") || !rel.endsWith("/possessions.json")) continue;
    const folder = rel.slice(0, rel.lastIndexOf("/"));
    const listed = new Set(index.possessions.map((row) => row.possession));
    if (listed.size !== index.possessions.length) problems.push(`${rel}: a possession is listed twice`);
    for (const number of listed) {
      if (!seen.has(`${folder}/${number}.json`)) problems.push(`${rel}: possession ${number} has no file`);
    }
    for (const other of seen) {
      if (!other.startsWith(`${folder}/`) || other === rel) continue;
      const number = Number(other.slice(folder.length + 1).replace(/\.json$/, ""));
      if (!listed.has(number)) problems.push(`${other}: not listed in ${rel}`);
    }
  }
  const players = documents.get("players.json");
  if (players) {
    const slugs = players.players.map((player) => player.slug);
    if (new Set(slugs).size !== slugs.length) problems.push("players.json: a player slug is listed twice");
    for (const slug of slugs) {
      if (!SLUG.test(slug)) problems.push(`players.json: player slug ${JSON.stringify(slug)} cannot be used in a URL`);
    }
  }
  return problems;
}

/** Validate the data root at `root`. Returns the list of problems, empty when the root is valid. */
export function validateTree(root, { allowMissing = [] } = {}) {
  const schema = loadSchema();
  const rules = loadRules();
  const notStaged = allowMissing.filter((name) => !rules.staged_files.includes(name));
  if (notStaged.length) {
    throw new Error(`only ${rules.staged_files.join(" and ")} may be missing, not ${notStaged.join(", ")}`);
  }
  const validate = validators(schema);
  const findPrivate = privateScanner(rules);
  const problems = [];
  const seen = new Set();
  const documents = new Map();
  for (const rel of listFiles(root, rules.ignored_names)) {
    seen.add(rel);
    const model = modelFor(schema, rel);
    if (!model) {
      problems.push(`${rel}: not a file the contract knows`);
      continue;
    }
    const text = readFileSync(path.join(root, rel), "utf8");
    for (const found of findPrivate(text)) problems.push(`${rel}: private string ${JSON.stringify(found)}`);
    let document;
    try {
      document = JSON.parse(text);
    } catch (error) {
      problems.push(`${rel}: not valid JSON: ${error.message}`);
      continue;
    }
    if (!validate[model](document)) {
      problems.push(`${rel}: ${model}: ${describeAjvError(validate[model].errors[0])}`);
      continue;
    }
    const parts = rel.split("/");
    if (model === "Possession" && (document.game_slug !== parts[1] || `${document.possession}.json` !== parts[2])) {
      problems.push(`${rel}: holds possession ${document.possession} of ${document.game_slug}`);
    }
    if (model === "PossessionIndex" && document.game.slug !== parts[1]) {
      problems.push(`${rel}: holds the index of ${document.game.slug}`);
    }
    documents.set(rel, document);
  }
  if (documents.has("manifest.json")) {
    problems.push(...manifestProblems(root, documents.get("manifest.json"), seen, schema));
  }
  for (const fixed of Object.keys(fixedFiles(schema))) {
    if (!seen.has(fixed) && !allowMissing.includes(fixed)) problems.push(`${fixed}: missing`);
  }
  problems.push(...siteProblems(documents, seen));
  return { problems, files: seen.size };
}

function allowMissingFrom(argv, env) {
  const flag = argv.indexOf("--allow-missing");
  const fromArgs = [];
  for (let index = flag + 1; flag >= 0 && index < argv.length && !argv[index].startsWith("--"); index += 1) {
    fromArgs.push(argv[index]);
  }
  const fromEnv = (env.SITE_DATA_ALLOW_MISSING ?? "").split(/[\s,]+/).filter(Boolean);
  return [...new Set([...fromArgs, ...fromEnv])];
}

export function main(argv = process.argv.slice(2), env = process.env) {
  const flag = argv.indexOf("--root");
  const root = flag >= 0 ? path.resolve(argv[flag + 1]) : dataRoot(env);
  if (!existsSync(root) || !statSync(root).isDirectory()) {
    console.error(`data: no data root at ${shown(root)}`);
    return 1;
  }
  const allowMissing = allowMissingFrom(argv, env);
  let result;
  try {
    result = validateTree(root, { allowMissing });
  } catch (error) {
    console.error(`data: ${error.message}`);
    return 2;
  }
  const { problems, files } = result;
  for (const line of problems.slice(0, 50)) console.error(`data: ${line}`);
  if (problems.length > 50) console.error(`data: ... and ${problems.length - 50} more`);
  let status = 0;
  if (problems.length) {
    console.error(`data: ${problems.length} problem(s) in ${shown(root)} (JSON Schema and files)`);
    status = 1;
  } else {
    console.log(`data: ${files} file(s) valid against the JSON Schema in ${shown(root)}`);
  }
  if (argv.includes("--python")) {
    const args = ["--validate", root];
    if (allowMissing.length) args.push("--allow-missing", ...allowMissing);
    let python;
    try {
      python = runPythonModule("export.schemas", args);
    } catch (error) {
      console.error(`data: ${error.message}`);
      return 1;
    }
    const lines = `${python.stdout}${python.stderr}`.split("\n").filter(Boolean);
    for (const line of lines) (python.status === 0 ? console.log : console.error)(`data: python: ${line}`);
    if (python.status !== 0) {
      console.error("data: python3 -m export.schemas --validate refused the data root (the contract's own rules)");
      status = 1;
    }
  }
  return status;
}

if (import.meta.url === `file://${process.argv[1]}`) process.exitCode = main();
