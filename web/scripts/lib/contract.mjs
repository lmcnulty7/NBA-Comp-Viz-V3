// The contract as the web side checks it: which model a file must match (a mirror of model_for in
// export/schemas.py), the JSON Schema validators (ajv) and the private-string scan (a mirror of find_private).
//
// The JSON Schema carries the shapes and the fixed labels. The cross-field rules of the contract live in the
// pydantic validators, so the full check of a data root is `python3 -m export.schemas --validate`, which
// `npm run data:validate` runs after the checks here.
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { CONTRACT_RULES_FILE, SCHEMA_FILE } from "./paths.mjs";

const require = createRequire(import.meta.url);
const Ajv2020 = require("ajv/dist/2020");

const SLUG = /^[a-z0-9][a-z0-9-]*$/;
const POSSESSION_FILE = /^\d+\.json$/;
// The machine-written texts (PLAN.md P3): the optional files of the genai/<kind>.json family, from GENAI_FILES of
// export/schemas.py (contract-rules.json).
let genaiFiles;

export function loadSchema(file = SCHEMA_FILE) {
  return JSON.parse(readFileSync(file, "utf8"));
}

/** The constants of export/schemas.py that the schema does not carry (written by `npm run types`). */
export function loadRules(file = CONTRACT_RULES_FILE) {
  return JSON.parse(readFileSync(file, "utf8"));
}

/** The fixed files of the contract (no <placeholder> in the path), as { path: model }. */
export function fixedFiles(schema) {
  return Object.fromEntries(Object.entries(schema["x-files"]).filter(([pattern]) => !pattern.includes("<")));
}

/** The model a file under the data root must validate against, or null for a path the contract does not know. */
export function modelFor(schema, relativePath) {
  const fixed = fixedFiles(schema);
  if (Object.hasOwn(fixed, relativePath)) return fixed[relativePath];
  const parts = relativePath.split("/");
  genaiFiles ??= Object.values(loadRules().genai_files ?? {});
  if (genaiFiles.includes(relativePath)) return schema["x-files"]["genai/<kind>.json"] ?? null;
  if (parts.length === 3 && parts[0] === "games" && SLUG.test(parts[1])) {
    if (parts[2] === "possessions.json") return "PossessionIndex";
    if (POSSESSION_FILE.test(parts[2])) return "Possession";
  }
  return null;
}

/** One ajv validator per file model. */
export function validators(schema) {
  const ajv = new Ajv2020({ allErrors: false, strict: true });
  ajv.addVocabulary(["x-schema-version", "x-compatible-versions", "x-files"]);
  ajv.addSchema(schema, "contract");
  const out = {};
  for (const model of new Set(Object.values(schema["x-files"]))) {
    out[model] = ajv.getSchema(`contract#/$defs/${model}`);
    if (!out[model]) throw new Error(`the schema has no definition for ${model}`);
  }
  return out;
}

export function describeAjvError(error) {
  const where = error.instancePath || "(root)";
  const extra = error.params?.additionalProperty ? ` "${error.params.additionalProperty}"` : "";
  return `${where} ${error.message}${extra}`;
}

// ------------------------------------------------------------------------------------------ private strings

// After a backslash, these keep their meaning in a JavaScript pattern with the u flag.
const KEPT_ESCAPES = new Set("dDsSbBnrtfv0123456789pPuxck^$\\.*+?()[]{}|/-".split(""));

/**
 * A Python `re` pattern as a JavaScript one with the u flag. Two things differ in the patterns of the contract:
 * Python's \w matches every Unicode letter and number (JavaScript's only ASCII; \p{L}\p{N}_ is the same set),
 * and Python allows a needless backslash before a quote, which the u flag refuses.
 */
export function toJsPattern(pythonPattern) {
  let out = "";
  let inClass = false;
  for (let index = 0; index < pythonPattern.length; index += 1) {
    const char = pythonPattern[index];
    if (char === "\\") {
      const next = pythonPattern[index + 1] ?? "";
      if (next === "w") out += inClass ? "\\p{L}\\p{N}_" : "[\\p{L}\\p{N}_]";
      else if (KEPT_ESCAPES.has(next)) out += char + next;
      else out += next;
      index += 1;
      continue;
    }
    if (char === "[") inClass = true;
    if (char === "]") inClass = false;
    out += char;
  }
  return out;
}

/** The scan of find_private in export/schemas.py: web addresses are dropped first, then the patterns are searched. */
export function privateScanner(rules = loadRules()) {
  const pattern = new RegExp(rules.private_patterns.map(toJsPattern).join("|"), "gu");
  const address = new RegExp(toJsPattern(rules.url_pattern), "gu");
  return (text) => {
    const plain = text.replaceAll("\\/", "/").replace(address, " ");
    const found = new Set();
    for (const match of plain.matchAll(pattern)) found.add(match[0]);
    // Python sorts strings by code point; so does this.
    return [...found].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  };
}
