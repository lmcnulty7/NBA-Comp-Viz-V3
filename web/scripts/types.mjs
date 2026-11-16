// The contract types (lib/types). Three generated files, never edited by hand:
//
//   contract.schema.json   written by `python3 -m export.schemas --out ...` (the JSON Schema of the contract)
//   contract.ts            TypeScript types, generated from the schema with json-schema-to-typescript
//   contract-rules.json    the constants of export/schemas.py that are not in the schema: the private-string
//                          patterns, the web-address filter, the ignored and the staged file names
//
//   node scripts/types.mjs --write   regenerate all three
//   node scripts/types.mjs --check   fail when any of them is stale
import { createRequire } from "node:module";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { CONTRACT_RULES_FILE, CONTRACT_TS_FILE, REPO_ROOT, SCHEMA_FILE, TYPES_DIR, shown } from "./lib/paths.mjs";
import { runPython, runPythonModule } from "./lib/python.mjs";

const require = createRequire(import.meta.url);
const { compile } = require("json-schema-to-typescript");

const BANNER = `/* eslint-disable */
/**
 * The site data contract as TypeScript types.
 *
 * GENERATED from lib/types/contract.schema.json, which export/schemas.py writes. Do not edit by hand:
 * change export/schemas.py, then run \`npm run types\`.
 */`;

/** pydantic gives every property a title, which the generator would turn into hundreds of one-line aliases.
 *  The names that matter are the keys under $defs, so every title is dropped before generating. */
function withoutTitles(node, isRoot = true) {
  if (Array.isArray(node)) return node.map((item) => withoutTitles(item, false));
  if (node && typeof node === "object") {
    const copy = {};
    for (const [key, value] of Object.entries(node)) {
      if (key === "title" && typeof value === "string" && !isRoot) continue;
      if (key === "$id" && isRoot) continue; // the root type is named by its title, not by the file name
      copy[key] = withoutTitles(value, false);
    }
    return copy;
  }
  return node;
}

export async function contractTs(schemaText) {
  const schema = JSON.parse(schemaText);
  const version = schema["x-schema-version"];
  const compatible = schema["x-compatible-versions"];
  const files = schema["x-files"];
  if (typeof version !== "string" || !files || typeof files !== "object" || !Array.isArray(compatible)) {
    throw new Error(
      "the schema has no x-schema-version, x-compatible-versions or x-files: was it written by export/schemas.py?",
    );
  }
  const body = await compile(withoutTitles(schema), "SiteContract", {
    bannerComment: BANNER,
    additionalProperties: false,
    unreachableDefinitions: true,
    strictIndexSignatures: false,
    format: true,
    style: { printWidth: 110, singleQuote: false, semi: true, trailingComma: "all" },
    cwd: TYPES_DIR,
  });
  const fileLines = Object.entries(files)
    .map(([pattern, model]) => `  ${JSON.stringify(pattern)}: ${JSON.stringify(model)},`)
    .join("\n");
  const footer = `
/** The schema version these types were generated from. The manifest carries the same value. */
export const SCHEMA_VERSION = ${JSON.stringify(version)} as const;

/** The versions a data file may carry: this one and the earlier minor versions of the same major version. */
export const COMPATIBLE_VERSIONS = ${JSON.stringify(compatible)} as const;

/** Path pattern under the data root, and the contract model each file must validate against. */
export const CONTRACT_FILES = {
${fileLines}
} as const;
`;
  return `${body.trimEnd()}\n${footer}`;
}

function pythonSchemaText() {
  const result = runPython([
    "-c",
    "import sys; from export import schemas as S; sys.stdout.write(S.schema_text())",
  ]);
  if (result.status !== 0) throw new Error(`python could not print the schema:\n${result.stderr}`);
  return result.stdout;
}

// The private-string scan of export/schemas.py is code: a list of patterns, a filter that drops web addresses
// first, and two lists of file names. They are copied out here so validate-data.mjs can do the same scan.
const RULES_SCRIPT = `
import json, sys
from export import schemas as S
rules = {
    "private_patterns": list(S.PRIVATE_PATTERNS),
    "url_pattern": S._URL_RE.pattern,
    "ignored_names": list(S.IGNORED_NAMES),
    "staged_files": list(S.STAGED_FILES),
    "fixture_source": S.FIXTURE_SOURCE,
}
sys.stdout.write(json.dumps(rules, indent=2) + "\\n")
`;

function pythonContractRules() {
  const result = runPython(["-c", RULES_SCRIPT]);
  if (result.status !== 0) throw new Error(`python could not print the contract rules:\n${result.stderr}`);
  return result.stdout;
}

function read(file) {
  return existsSync(file) ? readFileSync(file, "utf8") : "";
}

async function write() {
  mkdirSync(TYPES_DIR, { recursive: true });
  const schemaArg = path.relative(REPO_ROOT, SCHEMA_FILE);
  const result = runPythonModule("export.schemas", ["--out", schemaArg]);
  process.stdout.write(result.stdout);
  if (result.status !== 0) {
    process.stderr.write(result.stderr);
    throw new Error("python3 -m export.schemas --out failed");
  }
  writeFileSync(CONTRACT_TS_FILE, await contractTs(read(SCHEMA_FILE)));
  writeFileSync(CONTRACT_RULES_FILE, pythonContractRules());
  console.log(`types written: ${[SCHEMA_FILE, CONTRACT_TS_FILE, CONTRACT_RULES_FILE].map(shown).join(", ")}`);
}

async function check() {
  const problems = [];
  const schemaOnDisk = read(SCHEMA_FILE);
  if (schemaOnDisk !== pythonSchemaText()) {
    problems.push(`${shown(SCHEMA_FILE)} is not the schema export/schemas.py writes today`);
  }
  if (schemaOnDisk && read(CONTRACT_TS_FILE) !== (await contractTs(schemaOnDisk))) {
    problems.push(`${shown(CONTRACT_TS_FILE)} is not what the schema file generates`);
  }
  if (read(CONTRACT_RULES_FILE) !== pythonContractRules()) {
    problems.push(`${shown(CONTRACT_RULES_FILE)} is not what export/schemas.py states today`);
  }
  if (problems.length) {
    for (const line of problems) console.error(`stale: ${line}`);
    console.error("run `npm run types` in web/ and commit the result");
    return 1;
  }
  console.log("contract types are current (schema, TypeScript types, contract rules)");
  return 0;
}

const mode = process.argv[2];
if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    if (mode === "--write") {
      await write();
    } else if (mode === "--check") {
      process.exitCode = await check();
    } else {
      console.error("usage: node scripts/types.mjs --write | --check");
      process.exitCode = 2;
    }
  } catch (error) {
    console.error(`types: ${error.message}`);
    process.exitCode = 1;
  }
}
